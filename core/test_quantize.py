"""
Unit tests for core/quantize.py.

Checks:
1. Basic RTN correctness: quantized codes fall within the symmetric integer
   range [-(2^(d-1)-1), 2^(d-1)-1], and dequantized values reconstruct the
   input up to half a quantization step.
2. Monotonic accuracy vs bit-width: more bits -> lower reconstruction error.
3. Grouped quantization beats a single global scale when column-blocks have
   very different magnitudes (the whole point of group_size).
4. Group boundary handling when cols is not divisible by group_size.
5. clip_weights: clamp mode saturates correctly and leaves in-range values
   untouched; zero mode zeroes out-of-range values and leaves the rest.
6. search_p_clip finds a clipping threshold that reduces reconstruction
   error relative to naive (no-clip) quantization when a few extreme
   outliers dominate amax (the scenario clipping is meant to help).
7. All-zero input quantizes to all-zero output without division errors.
8. Output dtype/shape match input.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.quantize import (  # noqa: E402
    clip_and_quantize,
    clip_weights,
    quantize_symmetric,
    search_p_clip,
)


def test_codes_within_symmetric_range_and_reconstruct_reasonably():
    torch.manual_seed(0)
    R = torch.randn(16, 128) * 0.05
    bits = 4
    res = quantize_symmetric(R, bits=bits, group_size=None)
    qmax = (1 << (bits - 1)) - 1
    assert res.codes.abs().max().item() <= qmax

    # Reconstruction error per element should be within ~ half a step size.
    max_step = res.scale.max().item()
    err = (R - res.W_hat).abs().max().item()
    assert err <= max_step * 0.51 + 1e-6, (err, max_step)


def test_accuracy_improves_monotonically_with_bits():
    torch.manual_seed(1)
    R = torch.randn(32, 256) * 0.03
    errs = {}
    for bits in (2, 3, 4, 8):
        res = quantize_symmetric(R, bits=bits, group_size=128)
        errs[bits] = torch.linalg.matrix_norm(R - res.W_hat, ord="fro").item()
    bits_sorted = sorted(errs)
    for a, b in zip(bits_sorted, bits_sorted[1:]):
        assert errs[b] <= errs[a] + 1e-6, errs


def test_grouped_quantization_beats_global_scale_on_heterogeneous_columns():
    torch.manual_seed(2)
    rows = 8
    # Two column-blocks with very different magnitudes.
    block_small = torch.randn(rows, 128) * 0.01
    block_large = torch.randn(rows, 128) * 1.0
    R = torch.cat([block_small, block_large], dim=1)

    res_grouped = quantize_symmetric(R, bits=3, group_size=128)
    res_global = quantize_symmetric(R, bits=3, group_size=None)

    err_grouped = torch.linalg.matrix_norm(R - res_grouped.W_hat, ord="fro").item()
    err_global = torch.linalg.matrix_norm(R - res_global.W_hat, ord="fro").item()
    assert err_grouped < err_global, (err_grouped, err_global)


def test_group_boundary_handling_with_non_divisible_cols():
    torch.manual_seed(3)
    R = torch.randn(4, 130) * 0.02  # 130 = 128 + 2, not divisible by 128
    res = quantize_symmetric(R, bits=4, group_size=128)
    assert res.W_hat.shape == R.shape
    assert res.codes.shape == R.shape
    # 2 groups expected: one full (128), one partial (2).
    assert res.scale.shape[1] == 2


def test_clip_weights_clamp_mode():
    R = torch.tensor([[-5.0, -0.5, 0.0, 0.5, 5.0]])
    clipped = clip_weights(R, p_clip=1.0, mode="clamp")
    expected = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    assert torch.allclose(clipped, expected)


def test_clip_weights_zero_mode():
    R = torch.tensor([[-5.0, -0.5, 0.0, 0.5, 5.0]])
    clipped = clip_weights(R, p_clip=1.0, mode="zero")
    expected = torch.tensor([[0.0, -0.5, 0.0, 0.5, 0.0]])
    assert torch.allclose(clipped, expected)


def test_search_p_clip_helps_with_outliers():
    torch.manual_seed(4)
    rows, cols = 4, 128
    R = torch.randn(rows, cols) * 0.02
    # Inject a few extreme outliers that would otherwise blow up the scale.
    R[0, 0] = 5.0
    R[1, 10] = -4.0

    bits = 3
    naive = quantize_symmetric(R, bits=bits, group_size=128)
    naive_err = torch.linalg.matrix_norm(R - naive.W_hat, ord="fro").item()

    clipped_res = clip_and_quantize(R, bits=bits, group_size=128)
    clipped_err = torch.linalg.matrix_norm(R - clipped_res.W_hat, ord="fro").item()

    assert clipped_err < naive_err, (clipped_err, naive_err)


def test_search_p_clip_never_worse_than_no_clip_option():
    """Since percentiles includes 1.0 (= no clipping), the searched result
    should never be worse than plain quantize_symmetric."""
    torch.manual_seed(5)
    R = torch.randn(6, 128) * 0.04
    bits = 4
    naive = quantize_symmetric(R, bits=bits, group_size=128)
    naive_err = torch.linalg.matrix_norm(R - naive.W_hat, ord="fro").item()

    _, searched = search_p_clip(R, bits=bits, group_size=128)
    searched_err = torch.linalg.matrix_norm(R - searched.W_hat, ord="fro").item()

    assert searched_err <= naive_err + 1e-4, (searched_err, naive_err)


def test_all_zero_input():
    R = torch.zeros(4, 128)
    res = quantize_symmetric(R, bits=4, group_size=128)
    assert torch.all(res.W_hat == 0)
    assert torch.all(res.codes == 0)
    assert not torch.isnan(res.W_hat).any()


def test_output_dtype_and_shape_match_input():
    R = torch.randn(10, 200, dtype=torch.float16) * 0.02
    res = quantize_symmetric(R, bits=4, group_size=64)
    assert res.W_hat.dtype == torch.float16
    assert res.W_hat.shape == R.shape


if __name__ == "__main__":
    tests = [
        test_codes_within_symmetric_range_and_reconstruct_reasonably,
        test_accuracy_improves_monotonically_with_bits,
        test_grouped_quantization_beats_global_scale_on_heterogeneous_columns,
        test_group_boundary_handling_with_non_divisible_cols,
        test_clip_weights_clamp_mode,
        test_clip_weights_zero_mode,
        test_search_p_clip_helps_with_outliers,
        test_search_p_clip_never_worse_than_no_clip_option,
        test_all_zero_input,
        test_output_dtype_and_shape_match_input,
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