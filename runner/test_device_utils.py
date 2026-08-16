"""
Unit tests for runner/device_utils.py.

This sandbox has neither CUDA nor MPS available (see the module
docstring), so genuine cross-device generator recreation cannot be
exercised end-to-end here. These tests cover:
1. None-passthrough behavior (generator=None, device=None).
2. Same-device (CPU) no-op: the *same* generator object is returned, not
   a copy, when it's already on the requested device.
3. Device-string normalization: "cpu" (str) and torch.device("cpu")
   are treated as equal (no spurious recreation).
4. move_to_device passthrough for None tensor / None device.
5. move_to_device actually moves a tensor (trivially, cpu -> cpu, but
   exercises the real .to() call path).
6. resolve_device normalizes str/torch.device/None consistently.

For real cross-device (CPU -> MPS or CPU -> CUDA) verification, run this
manually on a machine with the relevant backend:

    generator = torch.Generator()
    generator.manual_seed(123)
    moved = resolve_generator_device(generator, "mps")  # or "cuda"
    assert moved is not generator
    assert moved.device.type == "mps"
    assert moved.initial_seed() == 123
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner.device_utils import (  # noqa: E402
    move_to_device,
    resolve_device,
    resolve_generator_device,
)


def test_resolve_generator_device_none_generator_passthrough():
    assert resolve_generator_device(None, "cpu") is None


def test_resolve_generator_device_none_device_passthrough():
    g = torch.Generator()
    g.manual_seed(1)
    assert resolve_generator_device(g, None) is g


def test_resolve_generator_device_same_device_returns_same_object():
    g = torch.Generator()
    g.manual_seed(7)
    result = resolve_generator_device(g, "cpu")
    assert result is g  # no unnecessary recreation


def test_resolve_generator_device_accepts_torch_device_object():
    g = torch.Generator()
    g.manual_seed(3)
    result = resolve_generator_device(g, torch.device("cpu"))
    assert result is g


def test_move_to_device_none_passthroughs():
    assert move_to_device(None, "cpu") is None
    t = torch.randn(3)
    assert move_to_device(t, None) is t


def test_move_to_device_moves_tensor():
    t = torch.randn(4, 4)
    moved = move_to_device(t, "cpu")
    assert torch.equal(moved, t)
    assert moved.device == torch.device("cpu")


def test_resolve_device_normalization():
    assert resolve_device(None) is None
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


if __name__ == "__main__":
    tests = [
        test_resolve_generator_device_none_generator_passthrough,
        test_resolve_generator_device_none_device_passthrough,
        test_resolve_generator_device_same_device_returns_same_object,
        test_resolve_generator_device_accepts_torch_device_object,
        test_move_to_device_none_passthroughs,
        test_move_to_device_moves_tensor,
        test_resolve_device_normalization,
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