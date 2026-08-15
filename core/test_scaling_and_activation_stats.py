"""
Unit tests for core/scaling.py and core/activation_stats.py.

Checks:
1. compute_alpha matches a hand-computed value on a small known X_bar.
2. compute_alpha never produces NaN/Inf, including on degenerate X_bar
   (all-equal channels, one dead channel).
3. apply_awq_scaling / unscale_r1flr_factors round-trip exactly (up to
   floating point) when the low-rank factorization is full-rank -- this is
   the key correctness check for the documented column-scaling /
   W_R-unscaling convention.
4. Scaling + R1-FLR + unscaling on a *reduced*-rank (lossy) factorization
   gives an equally good or comparable reconstruction to running R1-FLR
   directly on the unscaled matrix (sanity: scaling shouldn't wildly break
   things on an ordinary matrix without special channel structure).
5. ActivationStatsCollector.update is order-independent / incremental:
   feeding two chunks sequentially gives the same X_bar as feeding the
   concatenated batch at once.
6. ActivationStatsCollector.x_bar() matches a direct manual computation of
   "per-token normalized mean" on a synthetic tensor.
7. x_bar() raises before any update.
8. CalibrationHookManager captures the correct input activations from a
   real nn.Linear forward pass (integration test on a tiny model), with
   token counts matching what was actually fed through.
9. CalibrationHookManager respects the `layer_names` filter.
10. Hooks are properly removed on exit (no leftover hooks / stats stop
    accumulating after the context manager closes).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.activation_stats import ActivationStatsCollector, CalibrationHookManager  # noqa: E402
from core.r1_flr import r1_flr  # noqa: E402
from core.r1_sketch import r1_sketch_rank_r  # noqa: E402
from core.scaling import apply_awq_scaling, compute_alpha, unscale_r1flr_factors  # noqa: E402


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# scaling.py
# ---------------------------------------------------------------------------

def test_compute_alpha_matches_hand_computation():
    x_bar = torch.tensor([1.0, 2.0, 4.0])
    alpha = compute_alpha(x_bar)
    denom = (4.0 * 1.0) ** 0.5  # sqrt(max*min)
    expected = x_bar.pow(2.5) / denom
    assert torch.allclose(alpha, expected, atol=1e-6)


def test_compute_alpha_no_nan_on_degenerate_inputs():
    for x_bar in (
        torch.zeros(5),
        torch.ones(5),
        torch.tensor([0.0, 0.0, 3.0, 3.0]),
    ):
        alpha = compute_alpha(x_bar)
        assert not torch.isnan(alpha).any(), x_bar
        assert not torch.isinf(alpha).any(), x_bar
        assert (alpha > 0).all()


def test_scale_unscale_roundtrip_exact_at_full_rank():
    torch.manual_seed(0)
    m, n = 20, 16
    W = torch.randn(m, n)
    alpha = compute_alpha(torch.rand(n) + 0.1)  # positive, varied

    W_scaled = apply_awq_scaling(W, alpha)
    # Full-rank reconstruction of the *scaled* matrix.
    W_L, W_R_scaled = r1_sketch_rank_r(W_scaled, r=min(m, n), it=3, generator=_seeded_gen(0))
    _, W_R = unscale_r1flr_factors(W_L, W_R_scaled, alpha)

    recon = W_L @ W_R
    rel_err = (torch.linalg.matrix_norm(W - recon, ord="fro")
               / torch.linalg.matrix_norm(W, ord="fro")).item()
    assert rel_err < 1e-3, rel_err


def test_apply_scaling_shape_and_value_checks():
    W = torch.randn(5, 7)
    alpha = torch.rand(7) + 0.1
    W_scaled = apply_awq_scaling(W, alpha)
    assert W_scaled.shape == W.shape
    assert torch.allclose(W_scaled, W * alpha.unsqueeze(0), atol=1e-6)

    bad_alpha = torch.rand(3)  # wrong length
    try:
        apply_awq_scaling(W, bad_alpha)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_scaled_r1flr_pipeline_reconstruction_is_reasonable():
    """End-to-end: scale -> R1-FLR (lossy, flexible rank) -> unscale should
    give a reconstruction of comparable quality to running R1-FLR directly
    on the unscaled matrix (no special channel structure here, so scaling
    shouldn't help or hurt drastically)."""
    torch.manual_seed(1)
    m, n = 64, 48
    U_true = torch.randn(m, 6)
    V_true = torch.randn(6, n)
    W = U_true @ V_true + torch.randn(m, n) * 0.05

    x_bar = torch.rand(n) + 0.2
    alpha = compute_alpha(x_bar)

    W_scaled = apply_awq_scaling(W, alpha)
    res_scaled = r1_flr(W_scaled, d=4, dfp=16, x=0.5, t=0.0, it=2, generator=_seeded_gen(1))
    _, W_R_unscaled = unscale_r1flr_factors(res_scaled.W_L, res_scaled.W_R, alpha)
    recon_scaled_path = res_scaled.W_L @ W_R_unscaled
    err_scaled_path = torch.linalg.matrix_norm(W - recon_scaled_path, ord="fro").item()

    res_direct = r1_flr(W, d=4, dfp=16, x=0.5, t=0.0, it=2, generator=_seeded_gen(1))
    recon_direct = res_direct.W_L @ res_direct.W_R
    err_direct = torch.linalg.matrix_norm(W - recon_direct, ord="fro").item()

    # Not asserting one beats the other (that depends on channel structure
    # AWQ-style scaling is designed to exploit, which this synthetic matrix
    # doesn't have) -- just that the scaled pipeline is in the same
    # ballpark, i.e. the scale/unscale machinery isn't silently corrupting
    # the reconstruction.
    assert err_scaled_path < err_direct * 3 + 1e-6, (err_scaled_path, err_direct)


# ---------------------------------------------------------------------------
# activation_stats.py
# ---------------------------------------------------------------------------

def _manual_x_bar(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    X_abs = X.abs()
    row_mean = X_abs.mean(dim=1, keepdim=True).clamp_min(eps)
    normalized = X_abs / row_mean
    return normalized.mean(dim=0)


def test_collector_matches_manual_computation():
    torch.manual_seed(2)
    X = torch.randn(37, 12)
    expected = _manual_x_bar(X)

    collector = ActivationStatsCollector()
    collector.update(X)
    assert torch.allclose(collector.x_bar(), expected, atol=1e-5)


def test_collector_incremental_matches_batched():
    torch.manual_seed(3)
    X1 = torch.randn(10, 8)
    X2 = torch.randn(15, 8)
    X_all = torch.cat([X1, X2], dim=0)

    incremental = ActivationStatsCollector()
    incremental.update(X1)
    incremental.update(X2)

    batched = ActivationStatsCollector()
    batched.update(X_all)

    assert torch.allclose(incremental.x_bar(), batched.x_bar(), atol=1e-5)
    assert incremental.num_tokens_seen == batched.num_tokens_seen == 25


def test_collector_handles_leading_dims():
    """Activations typically arrive as (batch, seq, hidden); should flatten
    correctly to a token axis."""
    torch.manual_seed(4)
    X = torch.randn(2, 5, 8)  # batch=2, seq=5, hidden=8 -> 10 tokens
    collector = ActivationStatsCollector()
    collector.update(X)
    assert collector.num_tokens_seen == 10
    expected = _manual_x_bar(X.reshape(-1, 8))
    assert torch.allclose(collector.x_bar(), expected, atol=1e-5)


def test_x_bar_raises_before_update():
    collector = ActivationStatsCollector()
    try:
        collector.x_bar()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


class _TinyModel(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def test_hook_manager_captures_correct_inputs():
    torch.manual_seed(5)
    model = _TinyModel(hidden=8)
    x1 = torch.randn(3, 8)
    x2 = torch.randn(4, 8)

    with CalibrationHookManager(model) as mgr:
        model(x1)
        model(x2)
        x_bars = mgr.x_bars()

    assert set(x_bars.keys()) == {"fc1", "fc2"}
    # fc1's input is exactly x1 then x2 (concatenated).
    expected_fc1 = _manual_x_bar(torch.cat([x1, x2], dim=0))
    assert torch.allclose(x_bars["fc1"], expected_fc1, atol=1e-5)
    assert mgr.collectors["fc1"].num_tokens_seen == 7


def test_hook_manager_respects_layer_name_filter():
    model = _TinyModel(hidden=8)
    with CalibrationHookManager(model, layer_names={"fc1"}) as mgr:
        model(torch.randn(2, 8))
        collected = set(mgr.collectors.keys())
    assert collected == {"fc1"}


def test_hooks_removed_after_context_exit():
    model = _TinyModel(hidden=8)
    with CalibrationHookManager(model) as mgr:
        model(torch.randn(2, 8))
        tokens_before = mgr.collectors["fc1"].num_tokens_seen

    # Hooks should be gone now; running the model again must not change
    # the (now-frozen) collector state.
    model(torch.randn(5, 8))
    assert mgr.collectors["fc1"].num_tokens_seen == tokens_before


if __name__ == "__main__":
    tests = [
        test_compute_alpha_matches_hand_computation,
        test_compute_alpha_no_nan_on_degenerate_inputs,
        test_scale_unscale_roundtrip_exact_at_full_rank,
        test_apply_scaling_shape_and_value_checks,
        test_scaled_r1flr_pipeline_reconstruction_is_reasonable,
        test_collector_matches_manual_computation,
        test_collector_incremental_matches_batched,
        test_collector_handles_leading_dims,
        test_x_bar_raises_before_update,
        test_hook_manager_captures_correct_inputs,
        test_hook_manager_respects_layer_name_filter,
        test_hooks_removed_after_context_exit,
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