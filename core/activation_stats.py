"""
Activation capture for FLRQ's calibration-based scaling (Eq. 10-11).

Rather than caching every calibration activation tensor (expensive in
memory for a full model + 128x2048-token calibration set), this module
maintains a small running accumulator per `nn.Linear` layer and updates it
incrementally as calibration batches flow through the model via forward
pre-hooks. At the end of the calibration pass, each layer's accumulator
yields `X_bar`, the "per-token normalized mean of X" needed by
`scaling.compute_alpha`.

"per-token normalized mean": for a batch of activation vectors
X in R^(tokens, n), each token's vector is first divided by its own mean
absolute value (so a few unusually large-magnitude tokens don't dominate
the average -- consistent with the paper's framing of X_bar as a
*normalized* per-channel summary, not a raw mean), then the normalized
vectors are averaged across all tokens seen so far.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActivationStatsCollector:
    """Incrementally accumulates the per-input-channel normalized-mean
    activation statistic (X_bar) for a single layer, batch by batch."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._sum: torch.Tensor | None = None  # (n,), running sum of normalized rows
        self._count: int = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor) -> None:
        """Feed one batch of activations, shape (..., n) -- any leading
        dims (batch, sequence, etc.) are flattened into a token axis."""
        X = X.detach()
        n = X.shape[-1]
        X_flat = X.reshape(-1, n).to(torch.float32)

        X_abs = X_flat.abs()
        row_mean = X_abs.mean(dim=1, keepdim=True).clamp_min(self.eps)
        normalized = X_abs / row_mean  # (tokens, n)

        batch_sum = normalized.sum(dim=0)  # (n,)
        self._sum = batch_sum if self._sum is None else self._sum + batch_sum
        self._count += X_flat.shape[0]

    def x_bar(self) -> torch.Tensor:
        """Return the accumulated per-channel normalized mean, shape (n,)."""
        if self._count == 0 or self._sum is None:
            raise RuntimeError(
                "ActivationStatsCollector.x_bar() called with no activations "
                "collected yet -- run a calibration forward pass first."
            )
        return self._sum / self._count

    @property
    def num_tokens_seen(self) -> int:
        return self._count

    def reset(self) -> None:
        self._sum = None
        self._count = 0


class CalibrationHookManager:
    """Registers forward pre-hooks on a model's `nn.Linear` layers to
    capture each layer's *input* activations (the X in O = W @ X, i.e.
    the tensor actually fed into that Linear's forward call) via
    `ActivationStatsCollector`, and removes the hooks on exit.

    Usage:
        with CalibrationHookManager(model) as mgr:
            for batch in calibration_loader:
                model(batch)
        x_bar = mgr.collectors["decoder.layers.0.self_attn.q_proj"].x_bar()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: set[str] | None = None,
        eps: float = 1e-8,
    ):
        """
        Args:
            model: the model to instrument.
            layer_names: if given, only hook `nn.Linear` submodules whose
                fully-qualified name (as from `model.named_modules()`) is
                in this set. If None, hook every `nn.Linear`.
            eps: forwarded to each layer's `ActivationStatsCollector`.
        """
        self.model = model
        self.layer_names = layer_names
        self.eps = eps
        self.collectors: dict[str, ActivationStatsCollector] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "CalibrationHookManager":
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.layer_names is not None and name not in self.layer_names:
                continue
            collector = ActivationStatsCollector(eps=self.eps)
            self.collectors[name] = collector
            handle = module.register_forward_pre_hook(self._make_hook(collector))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

    @staticmethod
    def _make_hook(collector: ActivationStatsCollector):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            # nn.Linear's forward pre-hook receives (input,) as `inputs`.
            collector.update(inputs[0])
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def x_bars(self) -> dict[str, torch.Tensor]:
        """Return {layer_name: X_bar} for every layer with data collected."""
        return {
            name: collector.x_bar()
            for name, collector in self.collectors.items()
            if collector.num_tokens_seen > 0
        }