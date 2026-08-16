"""
Small device-handling utilities shared by the runner modules.

The one genuinely tricky part of "just move things to a device" in this
codebase is `torch.Generator`: PyTorch requires a generator passed to a
random-sampling op to live on the *same* device as the op's output tensor
(see `core/r1_sketch.py`, which calls `torch.randn(..., generator=...,
device=...)`). A plain `torch.Generator()` defaults to CPU, so a
`generator=` passed into `quantize_opt_model(..., device="mps")` (or
"cuda") would otherwise raise a device-mismatch RuntimeError deep inside
R1-Sketch. `resolve_generator_device` fixes that up front, preserving the
generator's seed so runs stay reproducible.

NOTE ON TEST COVERAGE: this sandboxed environment has neither a CUDA GPU
nor Apple Silicon/MPS available (verified: `torch.cuda.is_available()` and
`torch.backends.mps.is_available()` are both False here), so the actual
cross-device recreation path (cpu generator -> mps/cuda generator) cannot
be exercised end-to-end in this environment. It's implemented directly
against PyTorch's documented `torch.Generator(device=...)` /
`Generator.initial_seed()` API and covered here only for same-device
(CPU) no-ops and input validation; if you're running this on real
Apple Silicon or a GPU machine, `test_device_utils.py`'s
`test_manual_cross_device_check` has a commented-out snippet to verify the
real cross-device path there.
"""
from __future__ import annotations

import torch


def resolve_device(device: torch.device | str | None) -> torch.device | None:
    """Normalize a device argument (None passes through unchanged)."""
    if device is None:
        return None
    return torch.device(device)


def resolve_generator_device(
    generator: torch.Generator | None,
    device: torch.device | str | None,
) -> torch.Generator | None:
    """Return a generator guaranteed to live on `device`, preserving the
    original generator's seed.

    - If `generator` is None or `device` is None: returned unchanged
      (no-op passthrough).
    - If `generator` is already on `device`: returned unchanged (same
      object, not a copy).
    - Otherwise: a *new* `torch.Generator(device=device)` is created and
      seeded with `generator.initial_seed()`, so downstream sampling stays
      reproducible for a given seed regardless of which device it runs on.

    Raises:
        RuntimeError: if `device` doesn't support a torch.Generator at all
            in the installed PyTorch build (re-raised with a clearer
            message pointing at the cause).
    """
    if generator is None or device is None:
        return generator

    target = torch.device(device)
    current = torch.device(getattr(generator, "device", "cpu"))
    if current == target:
        return generator

    try:
        new_generator = torch.Generator(device=target)
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not create a torch.Generator on device '{target}' "
            "(this PyTorch build/backend may not support it). Pass "
            "generator=None to fall back to each device's default RNG "
            "stream instead of a seeded generator."
        ) from e
    new_generator.manual_seed(generator.initial_seed())
    return new_generator


def move_to_device(
    tensor: torch.Tensor | None,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    """`tensor.to(device)` that tolerates `tensor is None` or `device is
    None` (both pass through unchanged) -- convenience for optional
    arguments like `attention_mask`."""
    if tensor is None or device is None:
        return tensor
    return tensor.to(device)