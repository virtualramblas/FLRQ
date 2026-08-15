"""
Sequential, block-wise model quantization -- the architecture-agnostic
core of "Step G" (model integration) from the FLRQ implementation plan.

This is deliberately decoupled from OPT/transformers: it operates on any
ordered sequence of "blocks" (e.g. transformer decoder layers) given a
caller-supplied function to run one block's forward pass. The OPT-specific
plumbing (how to get that ordered list of blocks, how to call a block's
forward with the right kwargs, which of its submodules to quantize) lives
in `runner/opt_adapter.py`; this module is the reusable algorithm.

Algorithm (GPTQ/AWQ-style sequential calibration):
    hidden_states = <initial calibration activations, e.g. embeddings>
    for each block in order:
        1. Run this block's forward once on `hidden_states` with hooks
           attached to its target nn.Linear submodules, to capture each
           one's input activations (both the X_bar summary for
           AWQ-style scaling, and the raw activations for BLC's true
           objective).
        2. For each target nn.Linear in the block:
             a. alpha = compute_alpha(X_bar)
             b. run BLC on (alpha-scaled weight, raw activations) to get
                the best (W_L, W_R, W_q)
             c. un-scale W_R, replace the Linear's weight with the
                dequantized reconstruction W_L @ W_R + W_q (i.e. "fake
                quantization" -- the module stays a real nn.Linear so the
                model remains directly runnable/evaluable; packing to
                actual low-bit storage is a separate, later concern).
        3. Re-run the block's forward *once more*, now with its quantized
           weights, on `hidden_states` to obtain this block's output. That
           output becomes `hidden_states` for the *next* block -- so
           quantization error accumulates and is calibrated against
           realistically, rather than every block seeing pristine
           original activations (this is the same principle GPTQ/AWQ use
           for layer-wise calibration).

This keeps at most one block's calibration activations in memory at a
time (not the whole model's), which is what makes calibrating over e.g.
128x2048-token batches on a multi-billion-parameter model tractable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from core.activation_stats import CalibrationHookManager
from core.blc import blc
from core.scaling import apply_awq_scaling, compute_alpha, unscale_r1flr_factors

# A block-forward function takes (block_module, hidden_states, block_kwargs)
# and returns the block's output hidden_states tensor (leading dims
# preserved). `block_kwargs` is whatever extra state (attention mask,
# position ids, etc.) the block's forward needs; it is passed through
# unchanged on every call for a given block.
BlockForwardFn = Callable[[nn.Module, torch.Tensor, dict[str, Any]], torch.Tensor]


@dataclass
class LayerQuantStats:
    name: str
    rank: int
    bits: int
    in_features: int
    out_features: int
    best_epoch: int
    final_error: float
    extra_bit_fraction: float = field(init=False)

    def __post_init__(self) -> None:
        # matches the paper's "k" bookkeeping (Eq. 9): fraction of extra
        # storage the low-rank component adds relative to the base
        # quantized matrix, assuming dfp=16 for the stored factors.
        m, n, r, d = self.out_features, self.in_features, self.rank, self.bits
        self.extra_bit_fraction = (16 * r * (m + n)) / (d * m * n) if r > 0 else 0.0


@dataclass
class SequentialQuantResult:
    layer_stats: list[LayerQuantStats] = field(default_factory=list)

    def summary(self) -> str:
        if not self.layer_stats:
            return "no layers quantized"
        avg_rank = sum(s.rank for s in self.layer_stats) / len(self.layer_stats)
        avg_extra = sum(s.extra_bit_fraction for s in self.layer_stats) / len(self.layer_stats)
        return (
            f"{len(self.layer_stats)} layers quantized, "
            f"avg rank={avg_rank:.1f}, avg extra-bit fraction={avg_extra:.3f}"
        )


@torch.no_grad()
def quantize_linear_layer(
    linear: nn.Linear,
    X_bar: torch.Tensor,
    X_raw: torch.Tensor | None,
    bits: int,
    *,
    dfp: int = 16,
    x: float = 0.2,
    t: float = 1e-3,
    it: int = 2,
    group_size: int | None = 128,
    epochs: int = 20,
    generator: torch.Generator | None = None,
) -> LayerQuantStats:
    """Quantize a single nn.Linear in place: replaces `linear.weight.data`
    with the dequantized FLRQ reconstruction (scaling -> BLC -> unscale).

    Args:
        linear: the layer to quantize (modified in place).
        X_bar: per-input-channel normalized-mean activation summary,
            shape (in_features,) -- see activation_stats.py.
        X_raw: optional raw calibration activations, shape
            (num_tokens, in_features). If given, BLC uses the true
            objective E = ||WX - (Wr+Wq)X||; if None, BLC falls back to
            its weight-only proxy (see blc.py).
        bits, dfp, x, t, it, group_size, epochs, generator: forwarded to
            `blc()` / `r1_flr()`.

    Returns:
        LayerQuantStats describing what was chosen for this layer.
    """
    W = linear.weight.data  # (out_features, in_features)
    m, n = W.shape

    alpha = compute_alpha(X_bar.to(W.dtype))
    W_scaled = apply_awq_scaling(W, alpha)

    X_for_blc = X_raw.T.to(W.dtype) if X_raw is not None else None  # (in_features, tokens)

    result = blc(
        W_scaled, bits=bits, X=X_for_blc,
        dfp=dfp, x=x, t=t, it=it, group_size=group_size,
        epochs=epochs, generator=generator,
    )
    _, W_R_unscaled = unscale_r1flr_factors(result.W_L, result.W_R, alpha)
    # result.W_q was computed on the *scaled* residual (W_scaled - Wr_scaled);
    # since BLC's Wq is a per-group symmetric quantization of that residual
    # and the residual's *columns* were scaled by alpha the same way W was,
    # unscale it the same way as a weight matrix's columns would be.
    W_q_unscaled = result.W_q / alpha.to(W.dtype).unsqueeze(0)

    W_hat = result.W_L @ W_R_unscaled + W_q_unscaled
    linear.weight.data.copy_(W_hat.to(linear.weight.dtype))

    return LayerQuantStats(
        name="",  # filled in by caller with the qualified name
        rank=result.rank,
        bits=bits,
        in_features=n,
        out_features=m,
        best_epoch=result.best_epoch,
        final_error=min(result.error_curve) if result.error_curve else float("nan"),
    )


@torch.no_grad()
def quantize_blocks_sequential(
    blocks: Sequence[nn.Module],
    block_forward_fn: BlockForwardFn,
    block_kwargs: dict[str, Any],
    initial_hidden_states: torch.Tensor,
    linear_selector: Callable[[nn.Module], dict[str, nn.Linear]],
    bits: int,
    *,
    dfp: int = 16,
    x: float = 0.2,
    t: float = 1e-3,
    it: int = 2,
    group_size: int | None = 128,
    epochs: int = 20,
    generator: torch.Generator | None = None,
    block_name_prefix: str = "block",
) -> SequentialQuantResult:
    """Quantize an ordered sequence of blocks in place, propagating
    quantized-model activations from one block to the next.

    Args:
        blocks: ordered list of block modules (e.g. transformer decoder
            layers), already part of the live model (modified in place).
        block_forward_fn: callable(block, hidden_states, block_kwargs) ->
            hidden_states, running exactly one block's forward pass.
        block_kwargs: extra forward kwargs shared by every block (e.g.
            attention mask, position ids); passed through unchanged.
        initial_hidden_states: calibration activations to feed into the
            first block, shape (..., hidden_size).
        linear_selector: callable(block) -> {name: nn.Linear submodule}
            identifying which of a block's Linear layers to quantize.
        bits, dfp, x, t, it, group_size, epochs, generator: forwarded to
            `quantize_linear_layer` / `blc()` for every layer.
        block_name_prefix: used to build human-readable layer names in the
            returned stats (f"{block_name_prefix}{i}.{linear_name}").

    Returns:
        SequentialQuantResult with per-layer quantization stats, in the
        order layers were processed.
    """
    hidden_states = initial_hidden_states
    result = SequentialQuantResult()

    for block_idx, block in enumerate(blocks):
        targets = linear_selector(block)

        # 1. Calibration forward pass: capture each target Linear's input
        #    activations (both X_bar and, since we need BLC's true
        #    objective, the raw tokens) using the block's *current*
        #    (still fp/original) weights.
        target_names = set(targets.keys())
        with CalibrationHookManager(block, layer_names=target_names, keep_raw=True) as mgr:
            block_forward_fn(block, hidden_states, block_kwargs)

        # 2. Quantize each target layer in place.
        for name, linear in targets.items():
            collector = mgr.collectors[name]
            stats = quantize_linear_layer(
                linear,
                X_bar=collector.x_bar(),
                X_raw=collector.raw(),
                bits=bits,
                dfp=dfp, x=x, t=t, it=it,
                group_size=group_size, epochs=epochs,
                generator=generator,
            )
            stats.name = f"{block_name_prefix}{block_idx}.{name}"
            result.layer_stats.append(stats)

        # 3. Re-run the block with its now-quantized weights to get the
        #    input for the *next* block (sequential error propagation).
        hidden_states = block_forward_fn(block, hidden_states, block_kwargs)

    return result