"""
Unit tests for R1-FLR (Algorithm 1).

Checks:
1. amax curve is monotonically non-increasing as rank grows (Figure 2's
   basic shape requirement).
2. A tighter memory budget `x` yields rank <= a looser budget's rank on the
   same matrix (monotonic budget-vs-rank behaviour, Table 19's core trend).
3. x=0 (no headroom at all) accepts zero components -- boundary case of the
   `k > 1+x` criterion, since even a single rank-1 component's `k` exceeds
   `1 + 0 = 1`.
4. Reconstruction quality: FLRQ's picked rank should give reconstruction
   error at least as good as a fixed-rank sketch of *smaller* rank, and its
   own W_L @ W_R should match the residual actually consumed (self
   -consistency: W - W_L@W_R ≈ final residual amax value stored).
5. On an (approximately) low intrinsic-rank matrix, R1-FLR should stop well
   before max_rank (adaptivity -- it shouldn't just always run to cap).
6. max_rank hard cap is respected even if the size/precision criteria would
   otherwise keep accepting ranks.
7. All-zero input returns rank 0 without error.
8. Result is reproducible given a seeded generator.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.r1_flr import r1_flr  # noqa: E402


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make_decaying_matrix(m: int, n: int, seed: int, power: float = 1.2) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(m, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    decay = torch.tensor([1.0 / (i + 1) ** power for i in range(S.shape[0])])
    return (U * decay) @ Vh


def test_amax_curve_monotonically_non_increasing():
    # Strong spectral decay + a loose budget + a generous target bit-width
    # (larger d shrinks the size term k relative to r) -> several ranks
    # should be accepted, giving a meaningful curve to check monotonicity on.
    A = _make_decaying_matrix(64, 48, seed=0, power=1.6)
    res = r1_flr(A, d=8, dfp=16, x=2.0, t=0.0, it=2, generator=_seeded_gen(0))
    curve = res.amax_curve
    assert len(curve) >= 2, (res.rank, res.stop_reason, curve)
    for a, b in zip(curve, curve[1:]):
        assert b <= a + 1e-6, (curve,)


def test_tighter_budget_yields_smaller_or_equal_rank():
    A = _make_decaying_matrix(96, 64, seed=1)
    res_tight = r1_flr(A, d=4, dfp=16, x=0.05, t=0.0, it=2, generator=_seeded_gen(1))
    res_loose = r1_flr(A, d=4, dfp=16, x=0.5, t=0.0, it=2, generator=_seeded_gen(1))
    assert res_tight.rank <= res_loose.rank, (res_tight.rank, res_loose.rank)


def test_zero_budget_accepts_no_components():
    # With x=0, k > 1+x=1 for any rank r>=1, so no component can ever be
    # accepted. Note: the k>q check is evaluated first in the loop and may
    # also independently reject rank 1 -- either stop reason is fine, the
    # property under test is that zero headroom means zero accepted rank.
    A = _make_decaying_matrix(32, 32, seed=2)
    res = r1_flr(A, d=4, dfp=16, x=0.0, t=0.0, it=2, generator=_seeded_gen(2))
    assert res.rank == 0
    assert res.W_L.shape == (32, 0)
    assert res.W_R.shape == (0, 32)


def test_reconstruction_improves_with_accepted_rank():
    A = _make_decaying_matrix(80, 60, seed=3)
    res = r1_flr(A, d=3, dfp=16, x=0.3, t=0.0, it=2, generator=_seeded_gen(3))
    assert res.rank >= 1
    recon = res.W_L @ res.W_R
    full_err = torch.linalg.matrix_norm(A - recon, ord="fro").item()

    # A strictly smaller rank (drop the last accepted component) must not
    # reconstruct better than the full accepted set, since each component
    # was chosen to further reduce the residual.
    if res.rank >= 2:
        partial_recon = res.W_L[:, :-1] @ res.W_R[:-1, :]
        partial_err = torch.linalg.matrix_norm(A - partial_recon, ord="fro").item()
        assert full_err <= partial_err + 1e-4, (full_err, partial_err)

    # Self-consistency: the stored final amax_curve[-1] should equal the
    # amax of the actual residual (A - W_L@W_R), since that's exactly what
    # was measured to produce the stopping decision.
    actual_final_amax = torch.amax(torch.abs(A - recon)).item()
    assert abs(actual_final_amax - res.amax_curve[-1]) < 1e-3 * max(1.0, actual_final_amax)


def test_adaptivity_stops_before_max_rank_on_low_rank_matrix():
    torch.manual_seed(4)
    m, n, true_rank = 100, 80, 5
    U = torch.randn(m, true_rank)
    V = torch.randn(true_rank, n)
    A = U @ V  # exactly rank-5, so amax should collapse to ~0 quickly

    res = r1_flr(A, d=4, dfp=16, x=1.0, t=1e-4, it=2, max_rank=min(m, n),
                 generator=_seeded_gen(4))
    assert res.rank < min(m, n), "expected early stop on a low-rank matrix"
    assert res.rank <= true_rank + 3, (res.rank, true_rank)


def test_max_rank_hard_cap_respected():
    A = _make_decaying_matrix(50, 40, seed=5)
    res = r1_flr(A, d=8, dfp=16, x=10.0, t=0.0, it=2, max_rank=3,
                 generator=_seeded_gen(5))
    assert res.rank <= 3


def test_all_zero_matrix_returns_rank_zero():
    A = torch.zeros(10, 10)
    res = r1_flr(A, d=4, dfp=16, x=0.2, t=1e-3, it=2, generator=_seeded_gen(6))
    assert res.rank == 0
    assert "zero" in res.stop_reason


def test_reproducible_with_seeded_generator():
    A = _make_decaying_matrix(40, 30, seed=7)
    res1 = r1_flr(A, d=4, dfp=16, x=0.3, t=0.0, it=2, generator=_seeded_gen(42))
    res2 = r1_flr(A, d=4, dfp=16, x=0.3, t=0.0, it=2, generator=_seeded_gen(42))
    assert res1.rank == res2.rank
    assert torch.allclose(res1.W_L, res2.W_L)
    assert torch.allclose(res1.W_R, res2.W_R)


if __name__ == "__main__":
    tests = [
        test_amax_curve_monotonically_non_increasing,
        test_tighter_budget_yields_smaller_or_equal_rank,
        test_zero_budget_accepts_no_components,
        test_reconstruction_improves_with_accepted_rank,
        test_adaptivity_stops_before_max_rank_on_low_rank_matrix,
        test_max_rank_hard_cap_respected,
        test_all_zero_matrix_returns_rank_zero,
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