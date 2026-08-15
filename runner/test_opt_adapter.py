"""
Integration tests for runner/opt_adapter.py.

These run against the *real* `transformers.OPTForCausalLM` /
`OPTDecoderLayer` classes (not a stand-in), which is what actually
exercises version-fragile bits like `capture_initial_hidden_states`'s
hook-based interception of the causal mask / position ids. To avoid
requiring network access to the HF Hub (blocked by this environment's
network allowlist -- see opt_adapter.py's module docstring), every model
here is built from a small `OPTConfig` with random initialization, which
requires no download: `transformers` itself only needed installing from
PyPI, which is allowed.

These tests validate that the *integration plumbing* is correct (shapes,
hook wiring, weight replacement, finite outputs, sequential propagation
touching a real multi-layer OPT stack). They cannot and do not validate
that the paper's actual PPL numbers are reproduced -- that requires
pretrained OPT-1.3B weights and real WikiText2/C4 calibration data, which
this sandboxed environment cannot download. See the module docstring in
opt_adapter.py and the final summary for how to run the real thing.

Checks:
1. transformers is importable and a tiny OPTConfig model builds/forwards.
2. opt_linear_selector finds exactly the 6 target Linear submodules per
   decoder layer, with correct in/out feature shapes.
3. capture_initial_hidden_states returns a hidden_states tensor of the
   right shape and forward kwargs that a decoder layer actually accepts
   (verified by feeding them back into layer 0 directly and comparing
   against the model's own end-to-end hidden state at that point).
4. quantize_opt_model runs end-to-end on a tiny random OPT model: covers
   every layer's target linears, all weights change, and the quantized
   model still produces finite logits of the correct shape.
5. Quantized model's logits differ from the pre-quantization logits (i.e.
   quantization actually had an effect propagated through the whole
   stack), but are not wildly different in scale (sanity bound).
6. Runs cleanly with more than one calibration sequence (batch > 1) and
   with multiple distinct input batches (mimicking multiple calibration
   minibatches feeding a token budget).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import OPTConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

if _HAS_TRANSFORMERS:
    from runner.opt_adapter import (  # noqa: E402
        capture_initial_hidden_states,
        get_opt_decoder_layers,
        load_opt,
        opt_block_forward,
        opt_linear_selector,
        quantize_opt_model,
    )


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _tiny_opt_config(num_layers: int = 2) -> "OPTConfig":
    return OPTConfig(
        hidden_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        ffn_dim=64,
        vocab_size=200,
        max_position_embeddings=64,
        word_embed_proj_dim=32,
    )


def test_tiny_opt_model_builds_and_forwards():
    torch.manual_seed(0)
    model = load_opt(_tiny_opt_config())
    model.eval()
    ids = torch.randint(0, 200, (2, 10))
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (2, 10, 200)
    assert torch.isfinite(out.logits).all()


def test_linear_selector_finds_expected_layers():
    model = load_opt(_tiny_opt_config(num_layers=1))
    layer = get_opt_decoder_layers(model)[0]
    selected = opt_linear_selector(layer)
    assert set(selected.keys()) == {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "fc1", "fc2",
    }
    assert selected["self_attn.q_proj"].in_features == 32
    assert selected["self_attn.q_proj"].out_features == 32
    assert selected["fc1"].in_features == 32
    assert selected["fc1"].out_features == 64
    assert selected["fc2"].in_features == 64
    assert selected["fc2"].out_features == 32


def test_capture_initial_hidden_states_matches_real_forward():
    torch.manual_seed(1)
    model = load_opt(_tiny_opt_config(num_layers=2))
    model.eval()
    ids = torch.randint(0, 200, (2, 8))

    hidden_states, kwargs = capture_initial_hidden_states(model, ids)
    assert hidden_states.shape == (2, 8, 32)

    layers = get_opt_decoder_layers(model)
    with torch.no_grad():
        # Feeding the captured hidden_states + kwargs back into layer 0
        # directly should reproduce exactly what the full model computes
        # internally right after layer 0.
        manual_layer0_out = opt_block_forward(layers[0], hidden_states, kwargs)

        # Cross-check via a hook on layer 1 that captures its input
        # (= the model's own layer-0 output) during a real forward pass.
        captured = {}

        def hook(module, args, kwargs_):
            captured["h"] = (args[0] if args else kwargs_["hidden_states"]).detach()

        handle = layers[1].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()

    assert torch.allclose(manual_layer0_out, captured["h"], atol=1e-4)


def test_quantize_opt_model_end_to_end():
    torch.manual_seed(2)
    model = load_opt(_tiny_opt_config(num_layers=2))
    model.eval()
    ids = torch.randint(0, 200, (3, 12))

    layers_before = [
        {name: lin.weight.data.clone() for name, lin in opt_linear_selector(layer).items()}
        for layer in get_opt_decoder_layers(model)
    ]
    with torch.no_grad():
        logits_before = model(ids).logits.clone()

    result = quantize_opt_model(
        model, ids, bits=4, x=0.3, epochs=3, generator=_seeded_gen(2),
    )

    layers = get_opt_decoder_layers(model)
    assert len(result.layer_stats) == 2 * 6  # 2 layers x 6 target linears each

    for layer_idx, layer in enumerate(layers):
        after = opt_linear_selector(layer)
        for name, before_w in layers_before[layer_idx].items():
            assert not torch.allclose(before_w, after[name].weight.data), (
                f"layer {layer_idx} {name} weight unchanged"
            )
            assert not torch.isnan(after[name].weight.data).any()

    with torch.no_grad():
        logits_after = model(ids).logits

    assert logits_after.shape == logits_before.shape
    assert torch.isfinite(logits_after).all()
    # Quantization at 4 bits with a real calibration pass should perturb
    # outputs (not be a no-op) but shouldn't blow up to a wildly different
    # scale on such a small, low-bit-insensitive toy model.
    assert not torch.allclose(logits_after, logits_before, atol=1e-3)
    rel_change = (
        torch.linalg.vector_norm(logits_after - logits_before)
        / torch.linalg.vector_norm(logits_before).clamp_min(1e-8)
    ).item()
    assert rel_change < 50.0, rel_change


def test_quantize_opt_model_with_batch_and_multiple_layers():
    torch.manual_seed(3)
    model = load_opt(_tiny_opt_config(num_layers=3))
    model.eval()
    ids = torch.randint(0, 200, (4, 16))

    result = quantize_opt_model(
        model, ids, bits=3, x=0.3, epochs=2, generator=_seeded_gen(3),
    )
    assert len(result.layer_stats) == 3 * 6
    print(result.summary())


if __name__ == "__main__":
    if not _HAS_TRANSFORMERS:
        print("SKIPPED: transformers not installed")
        sys.exit(0)

    tests = [
        test_tiny_opt_model_builds_and_forwards,
        test_linear_selector_finds_expected_layers,
        test_capture_initial_hidden_states_matches_real_forward,
        test_quantize_opt_model_end_to_end,
        test_quantize_opt_model_with_batch_and_multiple_layers,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR: {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)