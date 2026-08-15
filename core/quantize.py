"""
Quantizer with (optional) weight clipping, used for the residual `Wq` half
of FLRQ's decomposition  W ~= W_L @ W_R + Wq  (Eq. 2 / Eq. 8).

Implements:
  - Grouped symmetric round-to-nearest (RTN) quantization (Eq. 8):
        s_r    = (2^(d-1) - 1) / amax(R)
        W_hat  = clamp(round(R / s_r), -(2^(d-1)-1), 2^(d-1)-1) * s_r
    applied per group of `group_size` contiguous columns (in-features),
    matching the paper's group_size=128 setup (aligned with AWQ/GPTQ
    convention).
  - Weight clipping: clamp R to [-p_clip, p_clip] *before* quantizing, so
    the scale is driven by p_clip instead of the true (possibly
    outlier-dominated) amax(R). This is the "LWC" (learnable weight
    clipping) step referenced in the paper's Figure 1 and used inside BLC
    (`Wclp = Clipping(W - Wr, p_clip)`, then `Wq = Quant(Wclp)`).

    NOTE on an ambiguity in the paper: the prose says clipping "sets a
    portion of the numbers with the largest absolute values to zero",
    which read literally would mean zeroing outliers rather than
    saturating them. Standard weight-clipping in the PTQ literature
    (OmniQuant's LWC, which Figure 1 explicitly names) instead *clamps*
    outliers to +/-p_clip and keeps them (just at reduced magnitude) so
    they still contribute post-quantization. We implement clamping as the
    default (`mode="clamp"`) since it matches the named LWC technique and
    is the standard/safer choice, but also expose `mode="zero"` for the
    literal reading so this can be A/B-tested during calibration against
    the paper's Table 9/10 numbers.
  - A simple grid search over p_clip (as a percentile of amax(R)) that
    minimizes a reconstruction objective -- used as a self-contained
    default. BLC (the next module) will override this with the true
    activation-aware objective E = ||WX - (Wr+Wq)X|| when calibration
    data is available; this module's search is what BLC falls back on
    for its first pass / for use without calibration data.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantResult:
    W_hat: torch.Tensor      # dequantized ("fake-quantized") weights, real-valued
    codes: torch.Tensor      # integer codes, same shape, dtype long
    scale: torch.Tensor      # per-group scales, shape (rows, n_groups)
    bits: int
    group_size: int | None


def _group_view(R: torch.Tensor, group_size: int | None) -> tuple[torch.Tensor, int, int]:
    """Reshape (rows, cols) -> (rows, n_groups, group_size) for grouped
    quantization. If cols is not divisible by group_size, the last group
    is smaller and handled by padding with zeros (excluded from amax via
    masking) then trimmed back on return.
    """
    rows, cols = R.shape
    if group_size is None or group_size >= cols:
        return R.unsqueeze(1), 1, cols  # single group covering all columns

    n_groups = (cols + group_size - 1) // group_size
    pad = n_groups * group_size - cols
    if pad:
        R = torch.cat([R, torch.zeros(rows, pad, dtype=R.dtype, device=R.device)], dim=1)
    return R.view(rows, n_groups, group_size), n_groups, cols


@torch.no_grad()
def quantize_symmetric(
    R: torch.Tensor,
    bits: int,
    group_size: int | None = 128,
) -> QuantResult:
    """Grouped symmetric RTN quantization (Eq. 8), no clipping.

    Args:
        R: matrix to quantize, shape (rows, cols).
        bits: target bit-width d (>= 2).
        group_size: number of contiguous columns sharing one scale. None
            (or >= cols) means a single per-row... actually per-*matrix*
            scale (one group spanning all columns).

    Returns:
        QuantResult with dequantized weights of the same shape/dtype as R.
    """
    if R.dim() != 2:
        raise ValueError(f"quantize_symmetric expects a 2-D matrix, got {tuple(R.shape)}")
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    rows, cols = R.shape
    dtype = R.dtype
    R_c = R.to(torch.float32)

    grouped, n_groups, orig_cols = _group_view(R_c, group_size)  # (rows, n_groups, gs)

    qmax = (1 << (bits - 1)) - 1  # e.g. bits=4 -> 7
    amax = grouped.abs().amax(dim=2, keepdim=True)  # (rows, n_groups, 1)
    # Avoid div-by-zero for all-zero groups: scale doesn't matter there
    # since codes will be 0 regardless.
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale = safe_amax / qmax  # s_r, shape (rows, n_groups, 1)

    codes = torch.clamp(torch.round(grouped / scale), -qmax, qmax)
    dequant = codes * scale

    dequant = dequant.reshape(rows, n_groups * (grouped.shape[2]))[:, :orig_cols]
    codes_out = codes.reshape(rows, n_groups * (grouped.shape[2]))[:, :orig_cols].to(torch.long)
    scale_out = scale.squeeze(2)  # (rows, n_groups)

    return QuantResult(
        W_hat=dequant.to(dtype),
        codes=codes_out,
        scale=scale_out.to(dtype),
        bits=bits,
        group_size=group_size,
    )


@torch.no_grad()
def clip_weights(
    R: torch.Tensor,
    p_clip: torch.Tensor | float,
    mode: str = "clamp",
) -> torch.Tensor:
    """Apply weight clipping at threshold `p_clip` (Wclp = Clipping(R, p_clip)).

    Args:
        R: matrix (or per-group-broadcastable tensor) to clip.
        p_clip: clipping threshold(s); scalar or broadcastable tensor
            (e.g. one value per group).
        mode: "clamp" (default) saturates |R| > p_clip to +/-p_clip
              (standard LWC-style clipping). "zero" instead sets those
              entries to 0 (the literal reading of the paper's prose --
              see module docstring). Both preserve everything with
              |R| <= p_clip unchanged.
    """
    if mode == "clamp":
        return torch.clamp(R, min=-p_clip, max=p_clip) if not torch.is_tensor(p_clip) \
            else torch.max(torch.min(R, p_clip), -p_clip)
    elif mode == "zero":
        if torch.is_tensor(p_clip):
            mask = R.abs() <= p_clip
        else:
            mask = R.abs() <= p_clip
        return R * mask
    else:
        raise ValueError(f"unknown clipping mode {mode!r}, expected 'clamp' or 'zero'")


@torch.no_grad()
def search_p_clip(
    R: torch.Tensor,
    bits: int,
    group_size: int | None = 128,
    percentiles: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5),
    mode: str = "clamp",
) -> tuple[torch.Tensor, QuantResult]:
    """Grid-search a per-group clipping threshold (expressed as a fraction
    of each group's amax) that minimizes Frobenius reconstruction error
    ||R - Clip+Quant(R)||_F, per group.

    This is a *weight-only* proxy objective (no activations), used as
    quantize.py's self-contained default and as BLC's fallback when no
    calibration input X is supplied for a given call. It mirrors the
    "search a clipping ratio per grid point" approach common to
    OmniQuant/AWQ-style calibration, simplified to a closed grid since we
    have no learnable parameters here.

    Returns:
        (p_clip, best_result): p_clip has shape (rows, n_groups, 1)
        (broadcastable against the grouped view), best_result is the
        QuantResult at the chosen threshold.
    """
    if R.dim() != 2:
        raise ValueError(f"search_p_clip expects a 2-D matrix, got {tuple(R.shape)}")

    rows, cols = R.shape
    R_c = R.to(torch.float32)
    grouped, n_groups, orig_cols = _group_view(R_c, group_size)
    group_amax = grouped.abs().amax(dim=2, keepdim=True)  # (rows, n_groups, 1)

    best_err = torch.full((rows, n_groups, 1), float("inf"))
    best_p = group_amax.clone()

    for pct in percentiles:
        p_clip = group_amax * pct
        clipped = clip_weights(grouped, p_clip, mode=mode)
        # quantize the clipped group directly (scale = p_clip/qmax, since
        # after clipping the group's own amax IS p_clip for pct<1, or the
        # true amax for pct==1).
        qmax = (1 << (bits - 1)) - 1
        safe_p = torch.where(p_clip > 0, p_clip, torch.ones_like(p_clip))
        scale = safe_p / qmax
        codes = torch.clamp(torch.round(clipped / scale), -qmax, qmax)
        dequant = codes * scale
        err = ((grouped - dequant) ** 2).sum(dim=2, keepdim=True)  # (rows, n_groups, 1)

        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_p = torch.where(improved, p_clip, best_p)

    # Final quantization at the chosen threshold.
    clipped = clip_weights(grouped, best_p, mode=mode)
    qmax = (1 << (bits - 1)) - 1
    safe_p = torch.where(best_p > 0, best_p, torch.ones_like(best_p))
    scale = safe_p / qmax
    codes = torch.clamp(torch.round(clipped / scale), -qmax, qmax)
    dequant = codes * scale

    dequant = dequant.reshape(rows, n_groups * grouped.shape[2])[:, :orig_cols]
    codes_out = codes.reshape(rows, n_groups * grouped.shape[2])[:, :orig_cols].to(torch.long)

    result = QuantResult(
        W_hat=dequant.to(R.dtype),
        codes=codes_out,
        scale=scale.squeeze(2).to(R.dtype),
        bits=bits,
        group_size=group_size,
    )
    return best_p, result


@torch.no_grad()
def clip_and_quantize(
    R: torch.Tensor,
    bits: int,
    group_size: int | None = 128,
    percentiles: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5),
    mode: str = "clamp",
) -> QuantResult:
    """Convenience wrapper: search for a good clipping threshold and
    return the resulting quantization, matching BLC's per-iteration step
    `Wq = Quant(Clipping(W - Wr, p_clip))`.
    """
    _, result = search_p_clip(R, bits, group_size=group_size,
                               percentiles=percentiles, mode=mode)
    return result