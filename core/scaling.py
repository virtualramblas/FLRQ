"""
Activation-aware scaling applied before R1-FLR low-rank extraction
(Eq. 10-11 of the FLRQ paper, "Low-rank quantization with calibration").

Idea (shared with AWQ): input channels with larger/more salient activation
magnitude matter more for the layer's output, so before extracting the
low-rank component we rescale the weight matrix's input-channel dimension
by a per-channel factor alpha derived from calibration activation
statistics, run R1-FLR on the *scaled* matrix, then undo the scaling on
the resulting factors so the low-rank product still equals (an
approximation of) the *original* weight.

    alpha = X_bar^2.5 / sqrt(max(X_bar) * min(X_bar))        (Eq. 11)
    {W_L, W_R'} = R1-FLR(W @ diag(alpha))                     (Eq. 10, adapted)
    W_R = W_R' @ diag(alpha)^-1

where X_bar is a per-input-channel summary of calibration activations
(see `activation_stats.py` for how it's collected): "per-token normalized
mean of X" -- each calibration token's activation vector is first divided
by its own mean absolute value (so tokens of very different overall scale
contribute equally), then averaged across tokens to give one value per
input channel.

IMPORTANT NOTE ON A NOTATIONAL AMBIGUITY IN THE PAPER:
Eq. 10 in the paper is written as `U = alpha^-1 U'`, i.e. it un-scales the
*left* factor U. That is only self-consistent if the paper's W has its
*input*-channel dimension indexed by rows (since alpha is a vector over
input channels, of length n = in_features, and U' as conventionally
defined is the (m, r) factor tied to W's row space). Throughout this
codebase (r1_sketch.py, r1_flr.py) we instead use the standard nn.Linear
convention: W has shape (m, n) = (out_features, in_features), so the
column space (dimension n) is the input-channel dimension, and R1-FLR's
right factor W_R (shape (r, n)) is the one tied to that space. Under our
convention, alpha (length n) must scale W's *columns*, and un-scaling
therefore must be applied to W_R, not W_L -- the literal `U = alpha^-1 U'`
substitution would silently produce a shape mismatch / wrong result here.

We implement the mathematically self-consistent version for our shape
convention (scale columns, un-scale W_R) rather than the literal notation,
and verify the round-trip is exact via `test_scaling.py`. This is flagged
as a deliberate, documented deviation to revisit if a reference
implementation surfaces.
"""
from __future__ import annotations

import torch


def compute_alpha(x_bar: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the per-channel scaling vector alpha from Eq. 11.

    Args:
        x_bar: per-input-channel activation summary, shape (n,), expected
            non-negative (it's built from absolute activation values --
            see `activation_stats.per_token_normalized_mean`).
        eps: numerical floor to avoid division by zero / zero-power issues
            when a channel is entirely inactive on the calibration set.

    Returns:
        alpha, shape (n,), all entries > 0.
    """
    if x_bar.dim() != 1:
        raise ValueError(f"x_bar must be 1-D (n,), got shape {tuple(x_bar.shape)}")
    x_bar_safe = x_bar.clamp_min(eps)
    numerator = x_bar_safe.pow(2.5)
    denom = torch.sqrt(x_bar_safe.max() * x_bar_safe.min()).clamp_min(eps)
    return numerator / denom


def apply_awq_scaling(W: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Scale W's input-channel (column) dimension by alpha:
    W_scaled[:, j] = alpha[j] * W[:, j].

    Args:
        W: weight matrix, shape (m, n) = (out_features, in_features).
        alpha: per-input-channel scale, shape (n,).
    """
    if W.dim() != 2:
        raise ValueError(f"W must be 2-D, got shape {tuple(W.shape)}")
    if alpha.dim() != 1 or alpha.shape[0] != W.shape[1]:
        raise ValueError(
            f"alpha must have shape (n,) matching W's in_features={W.shape[1]}, "
            f"got {tuple(alpha.shape)}"
        )
    return W * alpha.to(W.dtype).unsqueeze(0)


def unscale_r1flr_factors(
    W_L: torch.Tensor,
    W_R: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo `apply_awq_scaling` on R1-FLR's output factors so that
    W_L @ W_R approximates the *original*, unscaled W.

    Since only W's columns were scaled, W_L (tied to the row/output space)
    is returned unchanged; W_R (tied to the column/input space, shape
    (r, n)) has each column j divided by alpha[j].

    Args:
        W_L: left factor from R1-FLR run on the *scaled* matrix, shape (m, r).
        W_R: right factor from R1-FLR run on the *scaled* matrix, shape (r, n).
        alpha: the same per-channel scale passed to `apply_awq_scaling`,
            shape (n,).

    Returns:
        (W_L, W_R_unscaled) -- W_L is passed through unchanged (returned
        for a symmetric call signature / future-proofing), W_R_unscaled
        has shape (r, n).
    """
    if W_R.dim() != 2 or W_R.shape[1] != alpha.shape[0]:
        raise ValueError(
            f"W_R's column count ({W_R.shape[-1] if W_R.dim() == 2 else '?'}) "
            f"must match alpha's length ({alpha.shape[0]})"
        )
    alpha_safe = alpha.to(W_R.dtype).clamp_min(torch.finfo(W_R.dtype).eps if W_R.is_floating_point() else 1)
    W_R_unscaled = W_R / alpha_safe.unsqueeze(0)
    return W_L, W_R_unscaled