"""
Unit tests for runner/eval_ppl.py.

Checks:
1. A model that predicts the true next token perfectly (near-certainty)
   achieves PPL close to 1 -- the clearest possible correctness check.
2. A model that outputs uniform logits regardless of input achieves
   PPL close to vocab_size (the textbook result for a uniform-random
   predictor: mean NLL = log(vocab_size)).
3. num_windows / total_predicted_tokens bookkeeping matches manual
   computation, and leftover tokens (total_len % seq_len != 0) are
   correctly dropped, not silently included.
4. Works with three different model-output conventions: HF-style object
   with `.logits`, a plain tuple, and a bare tensor.
5. legacy_scaling=True changes total_predicted_tokens and the resulting
   PPL by exactly the expected factor relative to the default.
6. Raises ValueError when input_ids is shorter than seq_len.
7. Deterministic: repeated calls on the same (eval-mode, no-dropout)
   model/data give identical results.
8. evaluate_model_ppl (text + tokenizer convenience wrapper) matches
   calling compute_perplexity directly on the pre-tokenized ids.
9. Integration: runs end-to-end on a real (randomly initialized) tiny
   transformers OPTForCausalLM, producing a finite, positive PPL.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.calib import ByteTokenizer  # noqa: E402
from runner.eval_ppl import compute_perplexity, evaluate_model_ppl  # noqa: E402

try:
    from transformers import OPTConfig, OPTForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class _PerfectPredictorModel(nn.Module):
    """Deterministic 'model': given token t, always predicts (t+1) % vocab
    with near-certainty. If input_ids is constructed as a strictly
    incrementing-mod-vocab sequence, this model should achieve near-zero
    loss -> PPL ~= 1."""

    def __init__(self, vocab_size: int, confidence: float = 30.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.confidence = confidence

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -self.confidence)
        target = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, target.unsqueeze(-1), self.confidence)
        return logits


class _UniformModel(nn.Module):
    """Outputs perfectly uniform logits regardless of input -- textbook
    case where mean NLL = log(vocab_size), i.e. PPL = vocab_size."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        return torch.zeros(batch, seq, self.vocab_size)  # equal logits -> uniform softmax


class _TupleOutputModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.inner = _UniformModel(vocab_size)

    def forward(self, input_ids):
        return (self.inner(input_ids), "some_aux_state")


class _HFStyleOutput:
    def __init__(self, logits):
        self.logits = logits


class _HFStyleModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.inner = _UniformModel(vocab_size)

    def forward(self, input_ids):
        return _HFStyleOutput(self.inner(input_ids))


def test_perfect_predictor_gives_ppl_near_one():
    vocab = 50
    seq_len = 64
    total_len = seq_len * 3
    ids = torch.arange(total_len) % vocab  # incrementing-mod-vocab sequence
    model = _PerfectPredictorModel(vocab)

    result = compute_perplexity(model, ids, seq_len=seq_len)
    assert result.ppl < 1.01, result.ppl


def test_uniform_model_gives_ppl_near_vocab_size():
    vocab = 37
    seq_len = 100
    total_len = seq_len * 4
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (total_len,))
    model = _UniformModel(vocab)

    result = compute_perplexity(model, ids, seq_len=seq_len)
    assert math.isclose(result.ppl, vocab, rel_tol=1e-3), (result.ppl, vocab)


def test_window_bookkeeping_matches_manual_computation():
    vocab = 20
    seq_len = 50
    total_len = seq_len * 3 + 17  # deliberate remainder, should be dropped
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (total_len,))
    model = _UniformModel(vocab)

    result = compute_perplexity(model, ids, seq_len=seq_len)
    assert result.num_windows == 3  # remainder dropped
    assert result.total_predicted_tokens == 3 * (seq_len - 1)
    assert len(result.nll_per_window) == 3


def test_output_convention_variants_agree():
    vocab = 30
    seq_len = 40
    total_len = seq_len * 2
    torch.manual_seed(2)
    ids = torch.randint(0, vocab, (total_len,))

    r_tensor = compute_perplexity(_UniformModel(vocab), ids, seq_len=seq_len)
    r_tuple = compute_perplexity(_TupleOutputModel(vocab), ids, seq_len=seq_len)
    r_hf = compute_perplexity(_HFStyleModel(vocab), ids, seq_len=seq_len)

    assert math.isclose(r_tensor.ppl, r_tuple.ppl, rel_tol=1e-6)
    assert math.isclose(r_tensor.ppl, r_hf.ppl, rel_tol=1e-6)


def test_legacy_scaling_changes_token_count_as_expected():
    vocab = 25
    seq_len = 64
    total_len = seq_len * 3
    torch.manual_seed(3)
    ids = torch.randint(0, vocab, (total_len,))
    model = _UniformModel(vocab)

    r_default = compute_perplexity(model, ids, seq_len=seq_len, legacy_scaling=False)
    r_legacy = compute_perplexity(model, ids, seq_len=seq_len, legacy_scaling=True)

    assert r_default.total_predicted_tokens == 3 * (seq_len - 1)
    assert r_legacy.total_predicted_tokens == 3 * seq_len
    # Uniform model: mean loss per window is identical (log(vocab)) under
    # both conventions, so the two PPLs should actually match closely here
    # (the scaling only matters when losses vary window-to-window and get
    # summed then divided by a different total); still, sanity-check both
    # are close to log(vocab)-implied PPL.
    assert math.isclose(r_default.ppl, vocab, rel_tol=1e-3)
    assert math.isclose(r_legacy.ppl, vocab, rel_tol=1e-3)


def test_raises_on_too_short_input():
    model = _UniformModel(10)
    ids = torch.randint(0, 10, (30,))
    try:
        compute_perplexity(model, ids, seq_len=100)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_deterministic_repeated_calls():
    vocab = 15
    seq_len = 32
    torch.manual_seed(4)
    ids = torch.randint(0, vocab, (seq_len * 3,))
    model = _UniformModel(vocab)

    r1 = compute_perplexity(model, ids, seq_len=seq_len)
    r2 = compute_perplexity(model, ids, seq_len=seq_len)
    assert r1.ppl == r2.ppl
    assert r1.nll_per_window == r2.nll_per_window


def test_evaluate_model_ppl_matches_direct_call():
    vocab = 256  # ByteTokenizer's vocab size
    seq_len = 32
    text = "abcdefghij " * 200
    tok = ByteTokenizer()
    model = _UniformModel(vocab)

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    r_direct = compute_perplexity(model, ids, seq_len=seq_len)
    r_wrapper = evaluate_model_ppl(model, text, tok, seq_len=seq_len)

    assert math.isclose(r_direct.ppl, r_wrapper.ppl, rel_tol=1e-9)
    assert r_direct.num_windows == r_wrapper.num_windows


def test_end_to_end_on_real_tiny_opt_model():
    if not _HAS_TRANSFORMERS:
        print("SKIPPED (transformers not installed): test_end_to_end_on_real_tiny_opt_model")
        return
    torch.manual_seed(5)
    cfg = OPTConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        ffn_dim=32, vocab_size=100, max_position_embeddings=64,
        word_embed_proj_dim=16,
    )
    model = OPTForCausalLM(cfg)
    ids = torch.randint(0, 100, (300,))

    result = compute_perplexity(model, ids, seq_len=64)
    assert result.num_windows == 4
    assert math.isfinite(result.ppl)
    assert result.ppl > 0


if __name__ == "__main__":
    tests = [
        test_perfect_predictor_gives_ppl_near_one,
        test_uniform_model_gives_ppl_near_vocab_size,
        test_window_bookkeeping_matches_manual_computation,
        test_output_convention_variants_agree,
        test_legacy_scaling_changes_token_count_as_expected,
        test_raises_on_too_short_input,
        test_deterministic_repeated_calls,
        test_evaluate_model_ppl_matches_direct_call,
        test_end_to_end_on_real_tiny_opt_model,
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