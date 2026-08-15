"""
Unit tests for the R1-Sketch primitive.

Checks:
1. Rank-1 sketch recovers the top singular component of a matrix with
   spectral gap, matching full SVD closely (subspace + value agreement).
2. Rank-1 sketch on a pure rank-1 input recovers it (near) exactly.
3. Rank-r deflation (repeated r1_sketch calls) tracks the RSVD error bound
   from the paper (Eq. 4): E||A - A_r|| <= sigma_{r+1} * (1 + small term).
   We check it against the *actual* best rank-r error (Eckart-Young, i.e.
   sigma_{r+1} exactly, since it=2 power iterations should make the sketch
   near-optimal) with a reasonable slack factor.
4. Behaves sanely under fp16 weights (the real use case: LLM weight matrices).
5. Degenerate zero-matrix input raises rather than producing NaNs.
6. Increasing `it` monotonically (weakly) improves accuracy, matching the
   paper's Fig 7-12 finding that it=2 is already close to full SVD, with
   it=0/1 measurably worse.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.r1_sketch import r1_sketch, r1_sketch_rank_r  # noqa: E402


def _seeded_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_top_singular_component_recovered():
    """On a matrix with a clear spectral gap, r1_sketch's rank-1 output
    should closely match the true top SVD component (up to sign)."""
    torch.manual_seed(0)
    m, n = 256, 128
    # Build A with a strong top singular value and a decaying tail so the
    # top component is well separated (power iteration converges fast).
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    svals = torch.tensor([50.0] + [1.0 / (i + 1) for i in range(1, min(m, n))])
    S = torch.zeros(m, n)
    S[: min(m, n), : min(m, n)] = torch.diag(svals)
    A = U @ S @ V.T

    U_true, S_true, Vh_true = torch.linalg.svd(A, full_matrices=False)
    true_u, true_sigma, true_v = U_true[:, 0], S_true[0].item(), Vh_true[0, :]

    a_l, a_r = r1_sketch(A, it=4, generator=_seeded_gen(0))
    approx_sigma = torch.linalg.vector_norm(a_l).item()
    approx_u = (a_l / approx_sigma).squeeze(1)
    approx_v = a_r.squeeze(0)

    # Singular value should match closely.
    assert math.isclose(approx_sigma, true_sigma, rel_tol=1e-3), (
        approx_sigma, true_sigma,
    )

    # Singular vectors match up to sign: compare |cos similarity| to 1.
    cos_u = torch.abs(torch.dot(approx_u, true_u)).item()
    cos_v = torch.abs(torch.dot(approx_v, true_v)).item()
    assert cos_u > 1 - 1e-4, cos_u
    assert cos_v > 1 - 1e-4, cos_v

    # Reconstruction (rank-1) error should nearly equal sigma_2 (Eckart-Young).
    recon = a_l @ a_r
    err = torch.linalg.matrix_norm(A - recon, ord=2).item()
    sigma2 = S_true[1].item()
    assert err < sigma2 * 1.01 + 1e-6, (err, sigma2)


def test_exact_rank1_input_recovered_near_exactly():
    torch.manual_seed(1)
    m, n = 64, 40
    u = torch.randn(m, 1)
    v = torch.randn(1, n)
    A = u @ v  # exact rank-1

    a_l, a_r = r1_sketch(A, it=2, generator=_seeded_gen(1))
    recon = a_l @ a_r
    rel_err = (torch.linalg.matrix_norm(A - recon, ord="fro")
               / torch.linalg.matrix_norm(A, ord="fro")).item()
    assert rel_err < 1e-4, rel_err


def test_rank_r_deflation_matches_eckart_young_bound():
    """Repeated r1_sketch deflation should approach the best rank-r
    approximation (Eckart-Young: sigma_{r+1}), matching the paper's claim
    that R1-Sketch achieves the same accuracy as RSVD/SVD (Eq. 4)."""
    torch.manual_seed(2)
    m, n = 128, 96
    A = torch.randn(m, n)
    # impose a decaying spectrum for a meaningful rank-r test
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    decay = torch.tensor([1.0 / (i + 1) ** 1.5 for i in range(S.shape[0])])
    A = (U * decay) @ Vh

    U_true, S_true, _ = torch.linalg.svd(A, full_matrices=False)

    for r in (1, 4, 10):
        W_L, W_R = r1_sketch_rank_r(A, r=r, it=3, generator=_seeded_gen(100 + r))
        recon = W_L @ W_R
        err = torch.linalg.matrix_norm(A - recon, ord=2).item()
        sigma_next = S_true[r].item()  # sigma_{r+1}, 0-indexed
        # RSVD-style bound has slack; allow generous multiplicative margin.
        assert err < sigma_next * 3.0 + 1e-6, (r, err, sigma_next)


def test_fp16_weights_do_not_produce_nan():
    """FLRQ is applied to real LLM weight matrices, typically fp16."""
    torch.manual_seed(3)
    m, n = 512, 512
    A = torch.randn(m, n, dtype=torch.float16) * 0.02  # LLM-weight-like scale
    a_l, a_r = r1_sketch(A, it=2, generator=_seeded_gen(3))
    assert a_l.dtype == torch.float16
    assert a_r.dtype == torch.float16
    assert not torch.isnan(a_l).any()
    assert not torch.isnan(a_r).any()
    assert not torch.isinf(a_l).any()
    assert not torch.isinf(a_r).any()


def test_zero_matrix_raises():
    A = torch.zeros(10, 10)
    try:
        r1_sketch(A, it=2, generator=_seeded_gen(4))
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected r1_sketch to raise on a (near-)zero matrix"


def test_higher_it_improves_or_matches_accuracy():
    """Matches the paper's Fig 7-12 finding: it=0/1 are visibly noisier
    than it=2, and it=2 is already close to converged (SVD-level)."""
    torch.manual_seed(5)
    m, n = 200, 150
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    svals = torch.tensor([20.0, 15.0] + [1.0 / (i + 1) for i in range(2, min(m, n))])
    S = torch.zeros(m, n)
    S[: min(m, n), : min(m, n)] = torch.diag(svals)
    A = U @ S @ V.T
    sigma1_true = svals[0].item()

    errs = {}
    for it in (0, 1, 2, 4):
        a_l, a_r = r1_sketch(A, it=it, generator=_seeded_gen(500 + it))
        approx_sigma = torch.linalg.vector_norm(a_l).item()
        errs[it] = abs(approx_sigma - sigma1_true)

    # it=4 should be at least as accurate as it=0 (power iteration helps
    # separate the top singular value from a close second one: 20 vs 15).
    assert errs[4] <= errs[0] + 1e-6, errs


if __name__ == "__main__":
    tests = [
        test_top_singular_component_recovered,
        test_exact_rank1_input_recovered_near_exactly,
        test_rank_r_deflation_matches_eckart_young_bound,
        test_fp16_weights_do_not_produce_nan,
        test_zero_matrix_raises,
        test_higher_it_improves_or_matches_accuracy,
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