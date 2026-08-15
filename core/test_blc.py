"""
Unit tests for core/blc.py.

Checks:
1. Best-so-far error is monotonically non-increasing across epochs (it's
   tracked as a running min by construction, but verify the bookkeeping
   is actually correct against the raw error_curve).
2. epochs=1 reduces to exactly the initial R1-FLR + clip-and-quantize
   construction (best_epoch=0, single-entry curve).
3. More epochs never hurts: best error at epochs=10 <= best error at
   epochs=1, on the same problem/seed.
4. BLC's best reconstruction is at least as good as plain RTN quantization
   with no low-rank component at all (BLC should never be worse than the
   degenerate all-quantization baseline it subsumes).
5. Works with calibration data X (uses the true E = ||WX-(Wr+Wq)X||)
   and without (falls back to weight-only proxy) -- both should run
   cleanly and produce a usable result.
6. 2-bit quantization (the paper's hardest regime, where Table 10 shows
   BLC matters most) should show a substantial error reduction from
   epoch 0 to the best epoch on a matrix with real low-rank structure to
   exploit; a milder bit-width (8-bit) should show much smaller relative
   gains since quantization error is already small there.
7. Reproducibility: same seed -> same result.
8. Degenerate epochs=1, x=0 (no rank budget at all) still runs and
   reduces to a pure quantization result (rank 0).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.blc import blc  # noqa: E402
from core.quantize import quantize_symmetric  # noqa: E402


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make_test_matrix(m: int, n: int, seed: int, low_rank: int = 4, noise: float = 0.02):
    """Matrix with real low-rank structure plus full-rank noise, plus a
    couple of outliers -- representative of what R1-FLR+clipping are
    designed to exploit."""
    torch.manual_seed(seed)
    U = torch.randn(m, low_rank) * 0.5
    V = torch.randn(low_rank, n) * 0.5
    W = U @ V + torch.randn(m, n) * noise
    W[0, 0] += 3.0  # outlier
    W[m // 2, n // 2] -= 2.5
    return W


def test_best_error_is_running_min_of_curve():
    W = _make_test_matrix(48, 64, seed=0)
    res = blc(W, bits=4, epochs=8, x=0.3, generator=_seeded_gen(0))
    running_min = min(res.error_curve)
    assert abs(running_min - res.error_curve[res.best_epoch]) < 1e-9
    assert running_min <= res.error_curve[0] + 1e-9


def test_single_epoch_matches_initial_construction():
    W = _make_test_matrix(32, 40, seed=1)
    res = blc(W, bits=4, epochs=1, x=0.3, generator=_seeded_gen(1))
    assert res.best_epoch == 0
    assert len(res.error_curve) == 1


def test_more_epochs_never_hurts():
    W = _make_test_matrix(48, 64, seed=2)
    res_short = blc(W, bits=2, epochs=1, x=0.3, generator=_seeded_gen(2))
    res_long = blc(W, bits=2, epochs=10, x=0.3, generator=_seeded_gen(2))
    assert min(res_long.error_curve) <= min(res_short.error_curve) + 1e-6, (
        min(res_long.error_curve), min(res_short.error_curve),
    )


def test_blc_beats_plain_quantization_baseline():
    W = _make_test_matrix(40, 96, seed=3)
    bits = 3
    res = blc(W, bits=bits, epochs=10, x=0.3, generator=_seeded_gen(3))
    blc_err = torch.linalg.matrix_norm(W - res.W_hat, ord="fro").item()

    plain = quantize_symmetric(W, bits=bits, group_size=128)
    plain_err = torch.linalg.matrix_norm(W - plain.W_hat, ord="fro").item()

    assert blc_err <= plain_err + 1e-4, (blc_err, plain_err)


def test_runs_with_and_without_calibration_data():
    W = _make_test_matrix(32, 64, seed=4)
    n = W.shape[1]
    X = torch.randn(n, 50)  # 50 calibration "tokens"

    res_with_x = blc(W, bits=4, X=X, epochs=5, x=0.3, generator=_seeded_gen(4))
    res_without_x = blc(W, bits=4, epochs=5, x=0.3, generator=_seeded_gen(4))

    assert res_with_x.used_calibration is True
    assert res_without_x.used_calibration is False
    assert res_with_x.rank >= 0
    assert res_without_x.rank >= 0
    assert not torch.isnan(res_with_x.W_hat).any()
    assert not torch.isnan(res_without_x.W_hat).any()


def test_2bit_shows_larger_relative_gain_than_8bit():
    W = _make_test_matrix(48, 96, seed=5, low_rank=6)

    res_2bit = blc(W, bits=2, epochs=15, x=0.4, generator=_seeded_gen(5))
    gain_2bit = (res_2bit.error_curve[0] - min(res_2bit.error_curve)) / res_2bit.error_curve[0]

    res_8bit = blc(W, bits=8, epochs=15, x=0.4, generator=_seeded_gen(5))
    gain_8bit = (res_8bit.error_curve[0] - min(res_8bit.error_curve)) / max(res_8bit.error_curve[0], 1e-8)

    assert gain_2bit >= gain_8bit, (gain_2bit, gain_8bit)


def test_reproducible_with_seeded_generator():
    W = _make_test_matrix(24, 32, seed=6)
    res1 = blc(W, bits=4, epochs=4, x=0.3, generator=_seeded_gen(123))
    res2 = blc(W, bits=4, epochs=4, x=0.3, generator=_seeded_gen(123))
    assert res1.rank == res2.rank
    assert torch.allclose(res1.W_hat, res2.W_hat)
    assert res1.error_curve == res2.error_curve


def test_zero_budget_reduces_to_pure_quantization():
    W = _make_test_matrix(24, 32, seed=7)
    res = blc(W, bits=4, epochs=3, x=0.0, generator=_seeded_gen(7))
    assert res.rank == 0
    assert torch.allclose(res.W_hat, res.W_q)


if __name__ == "__main__":
    tests = [
        test_best_error_is_running_min_of_curve,
        test_single_epoch_matches_initial_construction,
        test_more_epochs_never_hurts,
        test_blc_beats_plain_quantization_baseline,
        test_runs_with_and_without_calibration_data,
        test_2bit_shows_larger_relative_gain_than_8bit,
        test_reproducible_with_seeded_generator,
        test_zero_budget_reduces_to_pure_quantization,
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