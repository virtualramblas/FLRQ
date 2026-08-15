"""
BLC: Best Low-rank Approximation under Clipping.

Implements Algorithm 2 of the FLRQ paper: an alternating-minimization outer
loop that iteratively refines the low-rank component `Wr` (via R1-FLR) and
the quantized residual `Wq` (via clip-and-quantize) to minimize

    min_{r, p_clip}  E[ ||W X - (Wr + Wq) X||_2 ]                 (Eq. 12)

where X is calibration-set activation data. Each iteration:
    1. Score the current (Wr, Wq) pair by the reconstruction error E.
    2. Recompute Wr = R1-FLR(W - Wq)   -- re-fit the low-rank part to the
       *current* quantization residual.
    3. Recompute Wq = Quant(Clip(W - Wr, p_clip))  -- re-fit the quantized
       part to the *new* low-rank residual, searching p_clip.
    4. Keep whichever (Wr, Wq) pair, across all epochs, achieved the
       lowest E.

This mirrors the paper's Algorithm 2 exactly: E is measured *before* the
epoch's updates, so the returned "best" result is always one that was
actually scored, not the (unscored) state after the final update.

Calibration data X: the paper defines E using per-layer activation inputs
captured from a calibration dataset (see the "Low-rank quantization with
calibration" section). This module accepts an optional `X` of shape
(n, num_calib_tokens) (n = W's input-feature dimension, i.e. W is assumed
to be a (m, n) = (out_features, in_features) matrix as in nn.Linear, and X
is calibration activations already transposed to features-first). If `X`
is omitted (e.g. for unit testing without a full model), BLC falls back to
a weight-only proxy objective ||W - (Wr+Wq)||_F -- this is *not* what the
paper measures, and is only intended for standalone testing / as a
degraded mode; the real per-layer integration (a later step) must supply
calibration activations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .quantize import clip_and_quantize
from .r1_flr import r1_flr


@dataclass
class BLCResult:
    W_L: torch.Tensor            # (m, rank) low-rank left factor, best epoch
    W_R: torch.Tensor            # (rank, n) low-rank right factor, best epoch
    W_q: torch.Tensor            # (m, n) dequantized quantized residual, best epoch
    rank: int
    error_curve: list[float] = field(default_factory=list)   # E at each epoch, pre-update
    rank_curve: list[int] = field(default_factory=list)      # rank chosen at each epoch's R1-FLR call
    best_epoch: int = 0
    used_calibration: bool = False

    @property
    def W_hat(self) -> torch.Tensor:
        """Full reconstruction: low-rank + quantized residual."""
        Wr = self.W_L @ self.W_R if self.rank > 0 else torch.zeros_like(self.W_q)
        return Wr + self.W_q


def _reconstruction_error(
    W: torch.Tensor,
    recon: torch.Tensor,
    X: torch.Tensor | None,
) -> float:
    """E = ||W X - recon X||_2 (Frobenius) if X given, else a weight-only
    Frobenius proxy ||W - recon||_F (see module docstring caveat)."""
    if X is not None:
        diff = (W - recon) @ X  # (m, n) @ (n, s) -> (m, s)
        return torch.linalg.matrix_norm(diff, ord="fro").item()
    return torch.linalg.matrix_norm(W - recon, ord="fro").item()


@torch.no_grad()
def blc(
    W: torch.Tensor,
    bits: int,
    X: torch.Tensor | None = None,
    *,
    dfp: int = 16,
    x: float = 0.2,
    t: float = 1e-3,
    it: int = 2,
    r1flr_max_rank: int | None = None,
    group_size: int | None = 128,
    clip_percentiles: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5),
    clip_mode: str = "clamp",
    epochs: int = 20,
    generator: torch.Generator | None = None,
) -> BLCResult:
    """Run BLC (Algorithm 2) on a single weight matrix.

    Args:
        W: weight matrix, shape (m, n) = (out_features, in_features).
        bits: target quantization bit-width d for the residual (also used
            by R1-FLR's rank-selection accounting).
        X: optional calibration activations, shape (n, num_calib_tokens).
           If None, falls back to a weight-only proxy objective (see
           module docstring).
        dfp, x, t, it, r1flr_max_rank: forwarded to `r1_flr` at each epoch.
        group_size, clip_percentiles, clip_mode: forwarded to
            `clip_and_quantize` at each epoch.
        epochs: number of BLC alternating-minimization iterations. The
            paper's own ablation (Table 22 / Fig 13) finds 3-/4-bit
            converge within ~1 epoch while 2-bit needs ~20.
        generator: optional torch.Generator, re-seeded identically at the
            start of every R1-FLR call within BLC for reproducibility
            (each call still uses fresh randomness *within* that call).

    Returns:
        BLCResult holding the (Wr, Wq) pair with the lowest observed E
        across all epochs, plus diagnostics.
    """
    if W.dim() != 2:
        raise ValueError(f"blc expects a 2-D matrix, got shape {tuple(W.shape)}")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    m, n = W.shape
    if X is not None and X.shape[0] != n:
        raise ValueError(
            f"X must have shape (n, num_calib_tokens) with n={n} matching "
            f"W's input dim, got X.shape={tuple(X.shape)}"
        )

    def _fresh_gen() -> torch.Generator | None:
        if generator is None:
            return None
        g = torch.Generator(device=generator.device if hasattr(generator, "device") else None)
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        return g

    def _run_r1_flr(residual: torch.Tensor):
        return r1_flr(
            residual, d=bits, dfp=dfp, x=x, t=t, it=it,
            max_rank=r1flr_max_rank, generator=_fresh_gen(),
        )

    # --- Init: Wr via R1-FLR on W itself, Wq = Quant(Clip(W - Wr)) ---
    init = _run_r1_flr(W)
    W_L, W_R = init.W_L, init.W_R
    rank = W_L.shape[1]
    Wr = W_L @ W_R if rank > 0 else torch.zeros_like(W)
    Wq_res = clip_and_quantize(
        W - Wr, bits=bits, group_size=group_size,
        percentiles=clip_percentiles, mode=clip_mode,
    )
    Wq = Wq_res.W_hat

    best_err = float("inf")
    best = (W_L, W_R, Wq, rank)
    best_epoch = 0
    error_curve: list[float] = []
    rank_curve: list[int] = [rank]

    for epoch in range(epochs):
        # 1. Score current (Wr, Wq).
        err = _reconstruction_error(W, Wr + Wq, X)
        error_curve.append(err)
        if err < best_err:
            best_err = err
            best = (W_L, W_R, Wq, rank)
            best_epoch = epoch

        # 2. Re-fit low-rank part to the current quantization residual.
        residual_after_quant = W - Wq
        flr_res = _run_r1_flr(residual_after_quant)
        W_L, W_R = flr_res.W_L, flr_res.W_R
        rank = W_L.shape[1]
        rank_curve.append(rank)
        Wr = W_L @ W_R if rank > 0 else torch.zeros_like(W)

        # 3. Re-fit the quantized residual to the new low-rank part.
        Wq_res = clip_and_quantize(
            W - Wr, bits=bits, group_size=group_size,
            percentiles=clip_percentiles, mode=clip_mode,
        )
        Wq = Wq_res.W_hat

    W_L_best, W_R_best, Wq_best, rank_best = best
    return BLCResult(
        W_L=W_L_best,
        W_R=W_R_best,
        W_q=Wq_best,
        rank=rank_best,
        error_curve=error_curve,
        rank_curve=rank_curve,
        best_epoch=best_epoch,
        used_calibration=X is not None,
    )