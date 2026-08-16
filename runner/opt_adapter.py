"""
OPT-specific adapter for the generic `quantize_blocks_sequential` engine.

Isolates everything that's specific to `transformers`' OPT implementation
(decoder layer list, which Linear submodules to target, and how to
correctly capture/replay a single decoder layer's forward pass) so that
`runner/quantize_model.py` stays architecture-agnostic.

IMPORTANT LIMITATION (network sandboxing): this environment's network
allowlist covers PyPI (so `transformers` itself installs fine) but not
huggingface.co, so pretrained OPT-1.3B weights (and the WikiText2/C4
calibration data used for the paper's Table 2 numbers) cannot be
downloaded here. This module is written and tested against the *real*
`OPTForCausalLM`/`OPTDecoderLayer` classes using a small, randomly
initialized config (no download required) to verify the integration
mechanics are correct. Running this against the real pretrained
OPT-1.3B (e.g. `load_opt("facebook/opt-1.3b")`) and real WikiText2
calibration data is expected to work unchanged in an environment with
Hub access; only weight/data acquisition is blocked here, not the code.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Submodule names (relative to an OPTDecoderLayer) that FLRQ targets, per
# Figure 1 of the paper (q/k/v/o projections in attention, fc1/fc2 in the
# MLP block).
OPT_LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
)


class _StopAtFirstLayer(Exception):
    def __init__(self, hidden_states: torch.Tensor, kwargs: dict[str, Any]):
        self.hidden_states = hidden_states
        self.kwargs = kwargs


def load_opt(name_or_config, **kwargs) -> nn.Module:
    """Load an OPT model. If `name_or_config` is a string, attempts to
    load pretrained weights via `AutoModelForCausalLM.from_pretrained`
    (requires Hub access). If it's an `OPTConfig` (or a dict of config
    kwargs), builds a randomly-initialized model with that architecture
    with no network access required -- useful for testing the
    quantization pipeline's mechanics without pretrained weights.
    """
    from transformers import AutoModelForCausalLM, OPTConfig, OPTForCausalLM

    if isinstance(name_or_config, str):
        return AutoModelForCausalLM.from_pretrained(name_or_config, **kwargs)
    if isinstance(name_or_config, OPTConfig):
        return OPTForCausalLM(name_or_config)
    if isinstance(name_or_config, dict):
        return OPTForCausalLM(OPTConfig(**name_or_config))
    raise TypeError(
        "name_or_config must be a HF model name/path (str), an OPTConfig, "
        f"or a dict of OPTConfig kwargs; got {type(name_or_config)}"
    )


def get_opt_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the ordered list of OPTDecoderLayer modules."""
    return list(model.model.decoder.layers)


def opt_linear_selector(layer: nn.Module) -> dict[str, nn.Linear]:
    """Resolve OPT_LINEAR_NAMES to actual nn.Linear submodules of a given
    decoder layer, skipping any that don't resolve (robustness against
    minor architecture variants)."""
    selected: dict[str, nn.Linear] = {}
    for name in OPT_LINEAR_NAMES:
        module = layer
        try:
            for part in name.split("."):
                module = getattr(module, part)
        except AttributeError:
            continue
        if isinstance(module, nn.Linear):
            selected[name] = module
    return selected


@torch.no_grad()
def capture_initial_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run a real forward pass and intercept it right before the first
    decoder layer, capturing the exact hidden_states and forward kwargs
    (causal mask, position_ids, etc.) that layer would receive.

    This deliberately reuses the model's own embedding/mask-construction
    logic (rather than reimplementing it) so it stays correct across
    `transformers` versions, at the cost of one extra (aborted) forward
    pass.
    """
    layers = get_opt_decoder_layers(model)
    if not layers:
        raise ValueError("model has no decoder layers")

    def hook(module: nn.Module, args: tuple, kwargs: dict[str, Any]):
        hidden_states = args[0] if args else kwargs.pop("hidden_states")
        rest = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise _StopAtFirstLayer(hidden_states.detach().clone(), rest)

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _StopAtFirstLayer as e:
        return e.hidden_states, e.kwargs
    finally:
        handle.remove()
    raise RuntimeError(
        "forward pre-hook on decoder layer 0 never fired -- model's forward "
        "signature may have changed; check OPTDecoder.forward in the "
        "installed transformers version."
    )


def opt_block_forward(block: nn.Module, hidden_states: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    """Run exactly one OPTDecoderLayer forward pass, normalizing across
    `transformers` versions that return either a bare tensor or a
    (hidden_states, ...) tuple."""
    out = block(hidden_states, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


def quantize_opt_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    bits: int,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    **quantize_kwargs: Any,
):
    """Quantize every decoder layer's q/k/v/o/fc1/fc2 projections in an
    OPT model, in place, using calibration data `input_ids`.

    Args:
        model: an OPTForCausalLM (or `.model` OPTModel) instance.
        input_ids: calibration token ids, shape (batch, seq_len).
        bits: target quantization bit-width.
        attention_mask: optional, forwarded to the initial embedding pass.
        device: if given, moves `model`, `input_ids`, and `attention_mask`
            there before quantizing (e.g. "cuda", "mps", "cpu"). If
            `quantize_kwargs` includes a `generator`, it is automatically
            re-created on `device` (preserving its seed) if it isn't
            already there -- see `runner/device_utils.py` for why this
            matters: a CPU `torch.Generator` cannot be used for sampling
            ops on a CUDA/MPS tensor and would otherwise raise a
            device-mismatch error deep inside R1-Sketch.
        **quantize_kwargs: forwarded to
            `runner.quantize_model.quantize_blocks_sequential` (dfp, x, t,
            it, group_size, epochs, generator, ...).

    Returns:
        SequentialQuantResult (see runner/quantize_model.py) with
        per-layer statistics.
    """
    from runner.device_utils import move_to_device, resolve_generator_device
    from runner.quantize_model import quantize_blocks_sequential

    if device is not None:
        model = model.to(device)
        input_ids = move_to_device(input_ids, device)
        attention_mask = move_to_device(attention_mask, device)
        if "generator" in quantize_kwargs:
            quantize_kwargs["generator"] = resolve_generator_device(
                quantize_kwargs["generator"], device
            )

    hidden_states, block_kwargs = capture_initial_hidden_states(
        model, input_ids, attention_mask=attention_mask
    )
    layers = get_opt_decoder_layers(model)

    return quantize_blocks_sequential(
        layers,
        opt_block_forward,
        block_kwargs=block_kwargs,
        initial_hidden_states=hidden_states,
        linear_selector=opt_linear_selector,
        bits=bits,
        block_name_prefix="decoder.layers.",
        **quantize_kwargs,
    )