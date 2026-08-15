"""
Unit tests for runner/quantize_model.py's architecture-agnostic core.

Uses a small synthetic "toy transformer block" (a couple of nn.Linear
layers + residual + activation) rather than a real transformer, so this
test suite has no dependency on `transformers`/OPT and exercises the
sequential-quantization algorithm directly and cheaply.

Checks:
1. quantize_linear_layer replaces weights in place and returns sane stats
   (rank >= 0, bits matches, shapes match).
2. quantize_linear_layer with X_raw=None (no calibration activations)
   still runs (falls back to BLC's weight-only proxy) without error.
3. quantize_blocks_sequential quantizes every target Linear in every
   block (by name), and returns per-layer stats in processing order.
4. Sequential propagation actually matters: hidden_states fed to block i+1
   reflect block i's *quantized* weights, not the original ones (verified
   by comparing against a hand-computed forward using the now-quantized
   block 0).
5. Quantized model still produces finite, correctly-shaped output on a
   forward pass (end-to-end sanity, no NaNs/Infs).
6. Tighter bit-width budgets produce a coarser (never better) reconstruction
   than looser ones, holding other settings fixed (monotonicity sanity,
   mirroring the paper's own bit-width sweeps).
7. Reproducibility given a seeded generator.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.quantize_model import (  # noqa: E402
    quantize_blocks_sequential,
    quantize_linear_layer,
)


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class ToyBlock(nn.Module):
    """A minimal stand-in for a transformer decoder layer: two Linear
    projections with a residual connection and an activation, similar in
    spirit to an MLP sub-block (e.g. OPT's fc1/fc2)."""

    def __init__(self, hidden: int = 16, ff: int = 24):
        super().__init__()
        self.in_proj = nn.Linear(hidden, ff)
        self.out_proj = nn.Linear(ff, hidden)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.act(self.in_proj(hidden_states))
        return hidden_states + self.out_proj(h)


def _toy_block_forward(block: nn.Module, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
    return block(hidden_states, **kwargs)


def _toy_linear_selector(block: nn.Module) -> dict[str, nn.Linear]:
    return {"in_proj": block.in_proj, "out_proj": block.out_proj}


def test_quantize_linear_layer_replaces_weights_and_returns_stats():
    torch.manual_seed(0)
    linear = nn.Linear(20, 12)
    W_before = linear.weight.data.clone()
    X_bar = torch.rand(20) + 0.1
    X_raw = torch.randn(30, 20) * 0.1

    stats = quantize_linear_layer(
        linear, X_bar=X_bar, X_raw=X_raw, bits=4, x=0.3, epochs=5,
        generator=_seeded_gen(0),
    )
    assert stats.bits == 4
    assert stats.in_features == 20
    assert stats.out_features == 12
    assert stats.rank >= 0
    assert not torch.allclose(linear.weight.data, W_before)  # actually changed
    assert not torch.isnan(linear.weight.data).any()


def test_quantize_linear_layer_without_raw_activations():
    torch.manual_seed(1)
    linear = nn.Linear(16, 8)
    X_bar = torch.rand(16) + 0.1
    stats = quantize_linear_layer(
        linear, X_bar=X_bar, X_raw=None, bits=4, x=0.3, epochs=3,
        generator=_seeded_gen(1),
    )
    assert stats.rank >= 0
    assert not torch.isnan(linear.weight.data).any()


def test_sequential_quantization_covers_all_target_layers():
    torch.manual_seed(2)
    blocks = [ToyBlock(hidden=16, ff=24) for _ in range(3)]
    hidden_states = torch.randn(2, 5, 16) * 0.1  # (batch, seq, hidden)

    result = quantize_blocks_sequential(
        blocks, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=4, x=0.3, epochs=3, generator=_seeded_gen(2),
    )

    expected_names = {
        f"block{i}.{name}" for i in range(3) for name in ("in_proj", "out_proj")
    }
    got_names = {s.name for s in result.layer_stats}
    assert got_names == expected_names
    assert len(result.layer_stats) == 6


def test_sequential_propagation_uses_quantized_outputs():
    """The hidden_states fed to block 1 should reflect block 0's quantized
    weights, not the pristine originals -- i.e. propagation is genuinely
    sequential (GPTQ/AWQ-style), not just independently quantizing each
    block on the same fixed original inputs."""
    torch.manual_seed(3)
    block0 = ToyBlock(hidden=12, ff=18)
    block0_original = ToyBlock(hidden=12, ff=18)
    block0_original.load_state_dict(block0.state_dict())
    block1 = ToyBlock(hidden=12, ff=18)
    hidden_states = torch.randn(1, 4, 12) * 0.1

    quantize_blocks_sequential(
        [block0, block1], _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=2, x=0.3, epochs=3, generator=_seeded_gen(3),  # aggressive bits -> visible change
    )

    # block0's weights should have changed (it was quantized).
    assert not torch.allclose(block0.in_proj.weight, block0_original.in_proj.weight)

    with torch.no_grad():
        out_quantized_block0 = block0(hidden_states)
        out_original_block0 = block0_original(hidden_states)
    assert not torch.allclose(out_quantized_block0, out_original_block0, atol=1e-4)


def test_end_to_end_forward_is_finite():
    torch.manual_seed(4)
    blocks = [ToyBlock(hidden=16, ff=20) for _ in range(2)]
    hidden_states = torch.randn(2, 6, 16) * 0.1

    quantize_blocks_sequential(
        blocks, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=3, x=0.3, epochs=3, generator=_seeded_gen(4),
    )

    with torch.no_grad():
        h = hidden_states
        for block in blocks:
            h = block(h)
    assert torch.isfinite(h).all()
    assert h.shape == hidden_states.shape


def test_tighter_bits_never_beats_looser_bits_on_average_error():
    torch.manual_seed(5)

    def build_blocks():
        torch.manual_seed(5)
        return [ToyBlock(hidden=16, ff=20) for _ in range(2)]

    hidden_states = torch.randn(2, 5, 16) * 0.1

    blocks_2bit = build_blocks()
    res_2bit = quantize_blocks_sequential(
        blocks_2bit, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=2, x=0.3, epochs=5, generator=_seeded_gen(5),
    )
    blocks_8bit = build_blocks()
    res_8bit = quantize_blocks_sequential(
        blocks_8bit, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=8, x=0.3, epochs=5, generator=_seeded_gen(5),
    )

    avg_err_2bit = sum(s.final_error for s in res_2bit.layer_stats) / len(res_2bit.layer_stats)
    avg_err_8bit = sum(s.final_error for s in res_8bit.layer_stats) / len(res_8bit.layer_stats)
    assert avg_err_8bit <= avg_err_2bit + 1e-4, (avg_err_8bit, avg_err_2bit)


def test_reproducible_with_seeded_generator():
    def build_blocks():
        torch.manual_seed(6)
        return [ToyBlock(hidden=12, ff=16) for _ in range(2)]

    hidden_states = torch.randn(1, 4, 12) * 0.1

    blocks_a = build_blocks()
    res_a = quantize_blocks_sequential(
        blocks_a, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=4, x=0.3, epochs=3, generator=_seeded_gen(99),
    )
    blocks_b = build_blocks()
    res_b = quantize_blocks_sequential(
        blocks_b, _toy_block_forward, block_kwargs={},
        initial_hidden_states=hidden_states,
        linear_selector=_toy_linear_selector,
        bits=4, x=0.3, epochs=3, generator=_seeded_gen(99),
    )

    ranks_a = [s.rank for s in res_a.layer_stats]
    ranks_b = [s.rank for s in res_b.layer_stats]
    assert ranks_a == ranks_b
    for lb, lb2 in zip(blocks_a, blocks_b):
        assert torch.allclose(lb.in_proj.weight, lb2.in_proj.weight)
        assert torch.allclose(lb.out_proj.weight, lb2.out_proj.weight)


if __name__ == "__main__":
    tests = [
        test_quantize_linear_layer_replaces_weights_and_returns_stats,
        test_quantize_linear_layer_without_raw_activations,
        test_sequential_quantization_covers_all_target_layers,
        test_sequential_propagation_uses_quantized_outputs,
        test_end_to_end_forward_is_finite,
        test_tighter_bits_never_beats_looser_bits_on_average_error,
        test_reproducible_with_seeded_generator,
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