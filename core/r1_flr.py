"""
R1-FLR: R1-Sketch-based Flexible Low-Rank Selection.

Implements Algorithm 1 of the FLRQ paper: starting from rank 0, repeatedly
extract a rank-1 component via `r1_sketch` on the current residual, and
decide -- after each extraction -- whether that additional rank was "worth
it" using a cheap proxy (the residual's max-abs value, `amax`) rather than
recomputing the true quantization error `E` at every rank (which would be
far more expensive; see Figure 2 / the paper's motivation section).

Stopping rule (paper's Eq. 9 and Algorithm 1): after extracting rank r,
compute
    p  = w0 / wr                      # amax reduction ratio, w0 = amax(W) before
                                       # any extraction, wr = amax(residual) at rank r
    d' = log2(p)                      # extra bits of precision this buys
    q  = (d + d') / d                 # effective-bit-improvement factor
    k  = 1 + dfp * r * (m + n) / (d * m * n)   # model-size-increase factor

and stop (rejecting this rank) if any of:
    - k > q          : the size increase outweighs the precision improvement
    - k > 1 + x       : the low-rank component alone would blow the memory
                         budget x (fraction of the base quantized model size)
    - slope(amax) < t : the amax curve has flattened out (diminishing returns)

Otherwise the rank-1 component is accepted and the loop continues.

Note on `t` (slope threshold) and the slope estimator itself: the paper
does not give a closed-form definition of `getSlope`. We implement it as
the (negative-signed, so "flattening" -> value near 0) relative decrease of
`amax` over a short trailing window, which is the natural reading of
Figure 2's use of a slope to detect when the curve has "leveled off". This
is documented as a deliberate design choice / ambiguity to revisit against
the paper's Table 9/10 numbers during calibration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .r1_sketch import r1_sketch


@dataclass
class R1FLRResult:
    """Diagnostics from an R1-FLR run, useful for both downstream BLC use
    and for reproducing the paper's Figure 2 / Table 9 style analyses."""

    W_L: torch.Tensor  # (m, rank)
    W_R: torch.Tensor  # (rank, n)
    rank: int
    amax_curve: list[float] = field(default_factory=list)  # amax after each accepted rank
    q_curve: list[float] = field(default_factory=list)
    k_curve: list[float] = field(default_factory=list)
    stop_reason: str = ""


def _slope(history: list[float], window: int) -> float:
    """Relative rate of decrease of `history` over the trailing `window`
    points: (oldest - newest) / (oldest * window), i.e. average fractional
    drop per step. Returns +inf if there isn't enough history yet (so the
    slope-based stopping criterion never fires prematurely), 0.0 if the
    tail is flat or increasing.
    """
    if len(history) < window + 1:
        return math.inf
    tail = history[-(window + 1):]
    oldest, newest = tail[0], tail[-1]
    if oldest <= 0:
        return 0.0
    drop = (oldest - newest) / oldest / window
    return max(drop, 0.0)


@torch.no_grad()
def r1_flr(
    W: torch.Tensor,
    d: int,
    dfp: int = 16,
    x: float = 0.2,
    t: float = 1e-3,
    it: int = 2,
    slope_window: int = 4,
    max_rank: int | None = None,
    generator: torch.Generator | None = None,
) -> R1FLRResult:
    """Flexible-rank low-rank selection for a single weight matrix.

    Args:
        W: weight matrix, shape (m, n).
        d: target quantization bit-width for the residual (used only in the
           rank-selection accounting, Eq. 9 -- R1-FLR itself does not
           quantize anything).
        dfp: bit-width of the *stored* low-rank factors (full precision,
             paper stores them losslessly -- default 16 for fp16).
        x: maximum allowed extra model-size fraction from the low-rank
           component (memory budget). Paper's ablation settles on x=0.2.
        t: slope threshold below which the amax curve is considered flat
           (diminishing returns) and extraction stops. See module docstring
           for the slope definition used here.
        it: power-iteration count passed to `r1_sketch` (paper finds it=2
            sufficient).
        slope_window: number of trailing ranks used to estimate the slope.
        max_rank: hard cap on rank (defaults to min(m, n)).
        generator: optional torch.Generator for reproducible sketching.

    Returns:
        R1FLRResult with the accepted low-rank factors and diagnostics.
    """
    if W.dim() != 2:
        raise ValueError(f"r1_flr expects a 2-D matrix, got shape {tuple(W.shape)}")
    if d <= 0 or dfp <= 0:
        raise ValueError("d and dfp must be positive bit-widths")

    m, n = W.shape
    cap = min(m, n) if max_rank is None else min(max_rank, min(m, n))

    residual = W.clone()
    w0 = torch.amax(torch.abs(residual)).item()

    left_cols: list[torch.Tensor] = []
    right_rows: list[torch.Tensor] = []
    amax_curve: list[float] = []
    q_curve: list[float] = []
    k_curve: list[float] = []
    stop_reason = f"reached max_rank={cap}"

    if w0 == 0.0:
        # W is already the zero matrix: nothing to extract.
        return R1FLRResult(
            W_L=torch.zeros(m, 0, dtype=W.dtype, device=W.device),
            W_R=torch.zeros(0, n, dtype=W.dtype, device=W.device),
            rank=0,
            stop_reason="input matrix is all-zero",
        )

    for r in range(1, cap + 1):
        try:
            u1, v1 = r1_sketch(residual, it=it, generator=generator)
        except ValueError:
            # Residual has numerically collapsed (e.g. we've already
            # extracted its full rank) -- nothing more to gain.
            stop_reason = "residual numerically exhausted"
            break

        residual = residual - u1 @ v1
        wr = torch.amax(torch.abs(residual)).item()

        # Guard against wr == 0 (perfect reconstruction at this rank):
        # treat as "infinite" precision improvement -> always accept, and
        # stop next iteration since amax_curve slope will be undefined.
        if wr <= 0.0:
            left_cols.append(u1)
            right_rows.append(v1)
            amax_curve.append(0.0)
            q_curve.append(math.inf)
            k_curve.append(1 + dfp * r * (m + n) / (d * m * n))
            stop_reason = "residual reached zero (exact reconstruction)"
            break

        p = w0 / wr
        d_prime = math.log2(p)
        q = (d + d_prime) / d
        k = 1 + dfp * r * (m + n) / (d * m * n)

        amax_curve.append(wr)
        q_curve.append(q)
        k_curve.append(k)

        slope_now = _slope(amax_curve, slope_window)

        if k > q:
            stop_reason = f"rank {r}: size-increase (k={k:.4f}) exceeds precision gain (q={q:.4f})"
            break
        if k > 1 + x:
            stop_reason = f"rank {r}: exceeded memory budget (k={k:.4f} > 1+x={1 + x:.4f})"
            break
        if slope_now < t:
            stop_reason = f"rank {r}: amax curve flattened (slope={slope_now:.6f} < t={t})"
            break

        left_cols.append(u1)
        right_rows.append(v1)

    if left_cols:
        W_L = torch.cat(left_cols, dim=1)
        W_R = torch.cat(right_rows, dim=0)
    else:
        W_L = torch.zeros(m, 0, dtype=W.dtype, device=W.device)
        W_R = torch.zeros(0, n, dtype=W.dtype, device=W.device)

    return R1FLRResult(
        W_L=W_L,
        W_R=W_R,
        rank=W_L.shape[1],
        amax_curve=amax_curve[: W_L.shape[1]],
        q_curve=q_curve[: W_L.shape[1]],
        k_curve=k_curve[: W_L.shape[1]],
        stop_reason=stop_reason,
    )