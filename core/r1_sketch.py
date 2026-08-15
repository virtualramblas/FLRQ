"""
R1-Sketch: Rank-1 randomized sketch under Gaussian projection.

This is the core primitive of FLRQ (Algorithm 1 / Eq. 5-7, 13-14 in the paper).
It is a specialization of Randomized SVD (RSVD) to the rank-1 case (a single
Gaussian test vector `s`), which lets the whole "Stage A / Stage B" RSVD
pipeline collapse into a couple of GEMV (matrix-vector) operations instead of
a QR decomposition + small SVD:

    P = (A A^T)^it A s                      # power iteration, GEMV only
    K = A^T P
    A_L = (||K|| / ||P||) * P / ||P||        # left singular-vector estimate (scaled)
    A_R = K / ||K||                          # right singular-vector estimate

The paper shows A_L @ A_R (outer product) approximates the rank-1 component
of A corresponding to its largest singular value, with the same accuracy /
error bound as full RSVD run in the rank-1 regime (Halko et al. 2011, Eq. 4
in the paper). Repeated calls on the deflated residual (A - A_L @ A_R) let
you build up a rank-r approximation one component at a time -- this is what
R1-FLR (the next module) does.

Complexity: computing P and K is O((2*it + 2) * m * n) (all GEMV), versus a
full SVD or even a batched RSVD-with-rank-r which needs GEMM + QR + a small
SVD. Norms ||P||, ||K|| are O(m) / O(n).
"""
from __future__ import annotations

import torch


@torch.no_grad()
def r1_sketch(
    A: torch.Tensor,
    it: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a rank-1 sketch approximation of `A` via Gaussian-projected
    power iteration.

    Args:
        A: matrix of shape (m, n) to approximate. Real-valued (fp32/fp16).
        it: number of power-iteration steps applied to (A A^T) before
            projecting through A. The paper finds it=2 is sufficient
            (Table 7 / Fig 7-12 ablation): higher it barely changes the
            result but costs more GEMVs.
        generator: optional torch.Generator for reproducible Gaussian
            sampling of the sketch vector `s`.

    Returns:
        (A_L, A_R): column vector of shape (m, 1) and row vector of shape
        (1, n) such that A_L @ A_R is the rank-1 approximation of A
        associated with (approximately) its largest singular value.
        A_L already carries the singular-value magnitude; A_R is a unit
        vector, matching the paper's convention (Eq. 14).

    Raises:
        ValueError: if A is not 2-D, or degenerates numerically (e.g. A is
            (numerically) the zero matrix), in which case P or K would be a
            zero vector and normalization is undefined.
    """
    if A.dim() != 2:
        raise ValueError(f"r1_sketch expects a 2-D matrix, got shape {tuple(A.shape)}")
    if it < 0:
        raise ValueError(f"it must be >= 0, got {it}")

    m, n = A.shape
    device, dtype = A.device, A.dtype

    # Do the numerically sensitive part (power iteration + normalization) in
    # fp32 regardless of the input dtype, then cast back at the end. Weight
    # matrices are often fp16 and repeated squaring of (A A^T) can otherwise
    # overflow/underflow.
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    A_c = A.to(compute_dtype)

    # 1. Gaussian test vector s in R^n.
    s = torch.randn(n, generator=generator, device=device, dtype=compute_dtype)

    # 2. Power iteration: P = (A A^T)^it A s, computed as repeated GEMVs
    #    (never materializing the m x m matrix A A^T).
    #
    #    Weight/residual matrices frequently have singular values < 1, so
    #    P's magnitude can shrink geometrically with `it` and underflow
    #    even though the *direction* has already converged. Since the
    #    final output only depends on the ratio ||K||/||P|| (see docstring
    #    derivation: rescaling P by any positive constant at any point
    #    rescales K by the same constant, leaving A_L invariant), it is
    #    safe -- and numerically necessary -- to renormalize p to unit
    #    norm after every application of (A A^T).
    eps = torch.finfo(compute_dtype).eps

    def _safe_normalize(v: torch.Tensor, what: str) -> torch.Tensor:
        v_norm = torch.linalg.vector_norm(v)
        if v_norm <= eps * max(m, n):
            raise ValueError(
                f"r1_sketch: ||{what}|| is numerically zero (matrix is ~0 "
                "or the test vector landed in the null space); cannot "
                "normalize."
            )
        return v / v_norm

    p = _safe_normalize(A_c @ s, "P")  # (m,), unit norm
    for _ in range(it):
        p = _safe_normalize(A_c @ (A_c.T @ p), "P")

    p_norm = torch.tensor(1.0, dtype=compute_dtype, device=device)

    # 3. K = A^T P  (this plays the role of B^* in the paper's Stage B, for
    #    the rank-1 case).
    k = A_c.T @ p  # (n,)
    k_norm = torch.linalg.vector_norm(k)
    if k_norm <= torch.finfo(compute_dtype).eps * max(m, n):
        raise ValueError(
            "r1_sketch: ||K|| is numerically zero; cannot normalize."
        )

    # 4. A_L = (||K||/||P||) * P/||P||  (shape (m,1)),  A_R = K/||K|| (shape (1,n))
    a_l = (k_norm / p_norm) * (p / p_norm)
    a_r = k / k_norm

    a_l = a_l.reshape(m, 1).to(dtype)
    a_r = a_r.reshape(1, n).to(dtype)
    return a_l, a_r


@torch.no_grad()
def r1_sketch_rank_r(
    A: torch.Tensor,
    r: int,
    it: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a rank-r approximation by repeatedly calling `r1_sketch` on the
    deflated residual. This is the naive fixed-rank building block that
    R1-FLR (Algorithm 1) wraps with a flexible stopping criterion; provided
    here mainly for testing r1_sketch's accuracy at rank > 1.

    Returns:
        (W_L, W_R) of shape (m, r) and (r, n) such that W_L @ W_R
        approximates A at rank r.
    """
    m, n = A.shape
    residual = A.clone()
    left_cols = []
    right_rows = []
    for _ in range(r):
        a_l, a_r = r1_sketch(residual, it=it, generator=generator)
        residual = residual - a_l @ a_r
        left_cols.append(a_l)
        right_rows.append(a_r)
    W_L = torch.cat(left_cols, dim=1)  # (m, r)
    W_R = torch.cat(right_rows, dim=0)  # (r, n)
    return W_L, W_R