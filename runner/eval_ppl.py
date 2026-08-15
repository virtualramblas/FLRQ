"""
Perplexity evaluation ("Step H" of the FLRQ implementation plan).

Implements the standard PTQ-literature evaluation protocol used by
GPTQ/AWQ/OmniQuant (and by the FLRQ paper itself, Table 2): concatenate
the test set's tokens into one long sequence, chop it into consecutive
non-overlapping windows of `seq_len` tokens (context length 2048 in the
paper), compute each window's next-token-prediction cross-entropy loss,
and combine into a single perplexity:

    PPL = exp( sum_i NLL_i / total_predicted_tokens )

where NLL_i is window i's total (not mean) negative log-likelihood.

A note on a common convention discrepancy: many published eval scripts
(including the original GPTQ repo this style descends from) scale each
window's mean loss by the *full* `seq_len` rather than `seq_len - 1` (the
actual number of next-token predictions once you shift by one position).
This is a systematic, ~1/seq_len-relative bias that mostly cancels out
when comparing different quantization methods evaluated with the same
script (as the paper does), but is not, strictly, the "true" perplexity.
This module computes the mathematically correct version
(`seq_len - 1` predicted tokens) by default, and exposes
`legacy_scaling=True` to reproduce the common (slightly biased)
convention exactly, for anyone trying to line up numbers with a published
table that used it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class PPLResult:
    ppl: float
    num_windows: int
    seq_len: int
    total_predicted_tokens: int
    nll_per_window: list[float] = field(default_factory=list)


def _extract_logits(model_output) -> torch.Tensor:
    """Unwrap a model's forward output into a plain logits tensor.
    Supports Hugging Face-style outputs with a `.logits` attribute,
    plain tuples (logits first), and bare tensors."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    seq_len: int = 2048,
    device: torch.device | str | None = None,
    legacy_scaling: bool = False,
) -> PPLResult:
    """Compute perplexity over consecutive non-overlapping windows of a
    (typically very long, e.g. full WikiText2-test) token sequence.

    Args:
        model: a causal language model. Its forward pass, given a batch of
            token ids, must return either an object with a `.logits`
            attribute (HF convention), a tuple with logits first, or a
            bare logits tensor of shape (batch, seq_len, vocab_size).
        input_ids: 1-D or (1, total_len) LongTensor of the full test-set
            token sequence (already concatenated, as in the paper's
            protocol -- not one document at a time).
        seq_len: window length (context length); the paper uses 2048.
        device: if given, both `model` and each batch are moved there.
        legacy_scaling: if True, scale each window's mean loss by the full
            `seq_len` (matching the common, slightly-biased GPTQ-style
            convention) instead of the mathematically correct
            `seq_len - 1` predicted tokens. See module docstring.

    Returns:
        PPLResult with the overall perplexity and per-window diagnostics.

    Raises:
        ValueError: if input_ids has fewer than `seq_len` tokens (can't
            form even one window).
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2 (need at least one prediction), got {seq_len}")

    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    ids = ids.to(torch.long)
    total_len = ids.shape[1]
    num_windows = total_len // seq_len
    if num_windows == 0:
        raise ValueError(
            f"input_ids has only {total_len} tokens, need at least "
            f"seq_len={seq_len} to form one evaluation window"
        )

    if device is not None:
        model = model.to(device)
    model.eval()

    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    nlls: list[float] = []
    tokens_per_window = seq_len if legacy_scaling else seq_len - 1

    for i in range(num_windows):
        batch = ids[:, i * seq_len:(i + 1) * seq_len]
        if device is not None:
            batch = batch.to(device)

        output = model(batch)
        logits = _extract_logits(output)

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()

        mean_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        window_nll = mean_loss.item() * tokens_per_window
        nlls.append(window_nll)

    total_predicted_tokens = num_windows * tokens_per_window
    ppl = math.exp(sum(nlls) / total_predicted_tokens)

    return PPLResult(
        ppl=ppl,
        num_windows=num_windows,
        seq_len=seq_len,
        total_predicted_tokens=total_predicted_tokens,
        nll_per_window=nlls,
    )


def evaluate_model_ppl(
    model: nn.Module,
    text: str,
    tokenizer,
    seq_len: int = 2048,
    device: torch.device | str | None = None,
    legacy_scaling: bool = False,
) -> PPLResult:
    """Convenience wrapper: tokenize raw text and evaluate perplexity in
    one call, matching the paper's "PPL on WikiText2/C4 test sets, context
    length 2048" workflow (Table 2).

    Args:
        model: causal LM, see `compute_perplexity`.
        text: raw test-set text (already concatenated across documents,
            as WikiText2/C4 test-set PPL evaluation conventionally does).
        tokenizer: any object with an `.encode(text) -> list[int]` method
            (a real HF tokenizer, or `data.calib.ByteTokenizer`).
        seq_len, device, legacy_scaling: forwarded to `compute_perplexity`.
    """
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return compute_perplexity(model, ids, seq_len=seq_len, device=device, legacy_scaling=legacy_scaling)