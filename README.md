# FLRQ: Flexible Low-Rank Quantization — PyTorch Implementation for OPT-1.3B

A from-scratch PyTorch implementation of **FLRQ** (Gu, Hu, Niu & Liu,
*"FLRQ: Faster LLM Quantization with Flexible Low-Rank Matrix Sketching"*,
AAAI 2026, [arXiv:2601.05684](https://arxiv.org/abs/2601.05684)), targeting
OPT-1.3B.

FLRQ combines two ideas to post-training-quantize LLM weights to low bit
widths (down to INT2) with minimal accuracy loss:

- **R1-FLR** (R1-Sketch-based Flexible Rank Selection): instead of a fixed
  low-rank correction applied uniformly to every layer, each layer gets an
  *adaptively chosen* rank, discovered cheaply via a rank-1 randomized-SVD
  sketch (`R1-Sketch`) applied one component at a time.
- **BLC** (Best Low-rank Approximation under Clipping): an alternating
  outer loop that iteratively refines the low-rank component and the
  quantized residual (with weight clipping) to minimize reconstruction
  error against real calibration activations.

This repo implements every stage of the paper's pipeline — the R1-Sketch
primitive, R1-FLR, the clipped quantizer, BLC, activation-aware (AWQ-style)
scaling, calibration data handling, sequential model integration for OPT,
and a perplexity evaluation harness — each as an independently unit-tested
module.

---

## Project structure

```
FLRQ/
├── core/
│   ├── r1_sketch.py          # Rank-1 randomized-SVD sketch 
│   ├── r1_flr.py              # Flexible rank selection loop 
│   ├── quantize.py            # Grouped symmetric RTN quantizer + clipping 
│   ├── blc.py                  # Best Low-rank approx. under Clipping 
│   ├── scaling.py              # AWQ-style activation-aware scaling 
│   └── activation_stats.py     # Forward-hook calibration activation capture
├── data/
│   └── calib.py                 # Calibration-set construction (128×2048-token sampling)
├── runner/
│   ├── quantize_model.py        # Architecture-agnostic sequential block quantizer
│   ├── opt_adapter.py            # OPT-specific wiring (transformers OPTForCausalLM)
│   ├── eval_ppl.py                # Sliding-window perplexity evaluation
│   └── device_utils.py             # device= / seeded-generator handling (CUDA/MPS/CPU)

```

Each `core/` module maps directly onto a named component/equation from the
paper. Each module has a matching `test_*.py` with a written explanation, in
its own docstring, of exactly what property is being checked and why.  

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # or OPTConfig for local testing
from data.calib import load_wikitext2_raw, build_calibration_set, CalibrationConfig
from runner.opt_adapter import quantize_opt_model
from runner.eval_ppl import evaluate_model_ppl

# Use an accelerator if available (CUDA, or MPS on Apple Silicon).
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model.eval()

# Paper's calibration protocol: 128 random 2048-token WikiText2 segments.
text = load_wikitext2_raw(split="train")  # or local_path="/path/to/wiki.train.raw"
calib_cfg = CalibrationConfig(num_samples=128, seq_len=2048, seed=0)
calib_ids = build_calibration_set(text, tokenizer, calib_cfg)

# Quantize every decoder layer's q/k/v/o/fc1/fc2 projections at 2 bits.
# `device=` moves the model, calibration ids, and (if given) a seeded
# `generator` there automatically -- see runner/device_utils.py.
result = quantize_opt_model(model, calib_ids, bits=2, x=0.2, epochs=20, device=device)
print(result.summary())

# Evaluate perplexity (context length 2048, matching Table 2).
test_text = load_wikitext2_raw(split="test")
ppl = evaluate_model_ppl(model, test_text, tokenizer, seq_len=2048, device=device)
print(f"WikiText2 PPL: {ppl.ppl:.2f}")
```


Both `quantize_opt_model` and `evaluate_model_ppl` accept a `device=`
argument (`"cuda"`, `"mps"`, `"cpu"`, or a `torch.device`). Passing it
moves the model and calibration/eval tensors there for you. 

You can also run the core logic by initializing a `OPTForCausalLM` class with a small, randomly
initialized config and a real (non-WikiText2) public-domain text corpus, as shown in the code example below:

```python
from transformers import OPTConfig, OPTForCausalLM
from data.calib import download_text_corpus, ByteTokenizer, build_calibration_set, CalibrationConfig
from runner.opt_adapter import quantize_opt_model
from runner.eval_ppl import evaluate_model_ppl

cfg = OPTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 ffn_dim=64, vocab_size=256, max_position_embeddings=512,
                 word_embed_proj_dim=32)
model = OPTForCausalLM(cfg).eval()

text = download_text_corpus("/tmp/demo_corpus.txt").read_text()  # real text, not WikiText2
tokenizer = ByteTokenizer()  # offline stand-in, not OPT's real BPE tokenizer

calib_ids = build_calibration_set(text, tokenizer, CalibrationConfig(num_samples=4, seq_len=128))
ppl_before = evaluate_model_ppl(model, text[:20000], tokenizer, seq_len=128)

result = quantize_opt_model(model, calib_ids, bits=4, x=0.3, epochs=3)
ppl_after = evaluate_model_ppl(model, text[:20000], tokenizer, seq_len=128)

print(ppl_before.ppl, "->", ppl_after.ppl)
# PPL before quantization: 256.256 over 156 windows
# 12 layers quantized, avg rank=0.1, avg extra-bit fraction=0.016
# PPL after quantization:  256.659 over 156 windows
```

(The pre-quantization PPL of ≈256 on a 256-token vocab with an untrained
model is exactly the textbook uniform-predictor result, `PPL = vocab_size`
— a good sanity signal that the evaluation harness is correct.)

## Running the unit tests

```bash
pip install torch transformers
cd FLRQ
for f in tests/test_*.py; do python3 "$f"; done
```

Each test file is also directly runnable and self-reporting (no `pytest`
dependency required, though it's compatible with `pytest` too).

| Module | Tests | What's specifically checked |
|---|---|---|
| `r1_sketch.py` | 6/6 | Matches full SVD's top singular component; exact recovery on rank-1 input; Eckart-Young bound at rank *r*; fp16 stability; degenerate-input handling; `it` sensitivity |
| `r1_flr.py` | 8/8 | Monotonic error curve; budget-vs-rank monotonicity; zero-budget boundary case; reconstruction improves with each accepted rank; adaptivity on low-rank inputs; hard rank cap; reproducibility |
| `quantize.py` | 11/11 | Code range and reconstruction bounds; monotonic accuracy vs. bit-width; grouped beats global scaling on heterogeneous columns; clipping helps with outliers; never worse than no-clip baseline; runs correctly on non-CPU devices |
| `blc.py` | 8/8 | Running-min bookkeeping; more epochs never hurts; beats plain RTN baseline; works with/without real calibration activations; **2-bit shows larger relative gain from BLC than 8-bit** (a direct check of the paper's own headline claim) |
| `scaling.py` + `activation_stats.py` | 14/14 | `α` formula correctness; exact scale/unscale round-trip at full rank; incremental activation accumulation matches batched computation; real hook capture on an `nn.Module` |
| `quantize_model.py` | 7/7 | Sequential *quantized*-activation propagation (not just per-layer isolated quantization); full model coverage; end-to-end finite outputs |
| `opt_adapter.py` + `device_utils.py` | 12/12 | Runs against the real `transformers.OPTForCausalLM`/`OPTDecoderLayer` classes; hook-based causal-mask/position-id capture verified against a real forward pass; `device=` moves model/ids/generator consistently |
| `calib.py` | 11/11 | Sampled segments are genuine contiguous substrings of the source (not just shape-correct noise); reproducibility; informative error when WikiText2 can't be fetched |
| `eval_ppl.py` | 9/9 | **Constructed ground-truth models**: a perfect next-token predictor gives PPL≈1; a uniform-logit model gives PPL≈vocab_size (textbook result) |

---

## Design decisions and documented ambiguities

The paper leaves a few implementation details underspecified or, in a
couple of places, internally inconsistent. Each is called out explicitly
in the relevant module's docstring; summarized here:

- **Weight clipping semantics** (`core/quantize.py`): the paper's prose
  ("setting a portion of the numbers with the largest absolute values to
  zero") reads as *zeroing* outliers, but Figure 1 labels the same step
  "LWC" (learnable weight clipping), the standard OmniQuant technique that
  *clamps* rather than zeros. We default to clamping (`mode="clamp"`,
  matching the named technique) and expose `mode="zero"` for the literal
  reading, so this is A/B-testable once real accuracy numbers are
  available.
- **Eq. 10's scaling-unscaling convention** (`core/scaling.py`): the
  paper writes `U = α⁻¹U'`, un-scaling the *left* factor. Since `α` is a
  vector over *input* channels and this codebase uses the standard
  `nn.Linear` convention (`W_L` tied to output/rows, `W_R` tied to
  input/columns), the literal substitution is a shape mismatch here. We
  implement the shape-consistent version (un-scale `W_R`) and verify it's
  an *exact* round-trip at full rank as evidence the convention is right.
- **The `getSlope` stopping criterion** (`core/r1_flr.py`): not given a
  closed form in the paper. Implemented as the average fractional drop in
  `amax` over a trailing window; flagged as a candidate for recalibration
  once real per-layer statistics (Table 9/10-style) are available.
- **BLC's true objective** (`core/blc.py`): the paper's `E =
  ||WX-(Wr+Wq)X||` needs real per-layer calibration activations. The
  standalone `blc()` function accepts optional `X` and uses it when given
  (this is what `runner/quantize_model.py` always supplies in the full
  pipeline); without it, `blc()` falls back to a documented weight-only
  proxy, used only for isolated module testing.
- **PPL scaling convention** (`runner/eval_ppl.py`): many published
  eval scripts scale each window's loss by the full `seq_len` rather than
  the true `seq_len - 1` predicted tokens (a small, mostly-cancelling
  bias). The mathematically correct version is the default;
  `legacy_scaling=True` reproduces the common convention exactly for
  side-by-side comparison with a published table.

## Requirements

```
torch
transformers   # only needed for OPT integration / real tokenizers
datasets    # only needed for OPT integration / real tokenizers
```

Everything in `core/` and `data/` has no dependency on `transformers` and
can be used/tested standalone.

## Citation

```bibtex
@inproceedings{gu2026flrq,
  title     = {FLRQ: Faster LLM Quantization with Flexible Low-Rank Matrix Sketching},
  author    = {Gu, Hongyaoxing and Hu, Lijuan and Niu, Shuzi and Liu, Fangfang},
  booktitle = {AAAI},
  year      = {2026}
}
```

## Disclaimer
This is an independent, unofficial reimplementation for research/learning
purposes and is not affiliated with the paper's authors.