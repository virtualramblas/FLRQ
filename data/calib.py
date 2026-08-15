"""
Calibration data loader ("Step A" of the FLRQ implementation plan).

Reproduces the paper's calibration protocol: 128 randomly selected
2048-token segments from WikiText2 (Merity et al. 2016), tokenized with
the model's own tokenizer -- "a good sampling strategy in OmniQuant"
(Setup, paper body).

NETWORK LIMITATION (same caveat as runner/opt_adapter.py): this sandbox's
network allowlist covers PyPI and GitHub (raw.githubusercontent.com,
codeload.github.com, github.com) but not huggingface.co, so the *actual*
WikiText2 dataset (normally fetched via `datasets.load_dataset` or the HF
Hub) and OPT's real BPE tokenizer (normally fetched via
`AutoTokenizer.from_pretrained`) cannot be downloaded here. This module is
written so the real path (`load_wikitext2_raw` + a real HF tokenizer) will
work unchanged in an environment with Hub access, and additionally
provides two offline-friendly fallbacks for development/testing without
that access:
  - `download_text_corpus`: fetches a small public-domain text file over
    an allowlisted domain (default: the "tinyshakespeare" corpus from
    a public GitHub repo) as a stand-in with *real* text structure --
    explicitly NOT WikiText2, only useful for exercising the pipeline.
  - `ByteTokenizer`: a dependency-free UTF-8 byte-level tokenizer (vocab
    size 256) requiring no downloaded vocab/merge files, so the
    segment-sampling logic can be tested fully offline on real text.
"""
from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import torch


@dataclass
class CalibrationConfig:
    """Matches the paper's Setup section: 128 segments of 2048 tokens."""
    num_samples: int = 128
    seq_len: int = 2048
    seed: int = 0


# ---------------------------------------------------------------------------
# Tokenizer interface + offline fallback
# ---------------------------------------------------------------------------

class Tokenizer(Protocol):
    """Minimal interface this module needs from a tokenizer -- satisfied
    directly by a real Hugging Face `PreTrainedTokenizer` (its `.encode`
    method matches this signature) as well as by `ByteTokenizer` below."""

    def encode(self, text: str) -> list[int]:
        ...


class ByteTokenizer:
    """Dependency-free, fully offline UTF-8 byte-level tokenizer.

    NOT what real FLRQ experiments should use -- the paper tokenizes with
    each model's own (sub-word) tokenizer, e.g. OPT's BPE tokenizer loaded
    via `AutoTokenizer.from_pretrained("facebook/opt-1.3b")`, which
    requires Hub network access unavailable in this sandbox. This class
    exists purely so calibration-loading logic (segment sampling, batching)
    can be exercised end-to-end on real text without any network or model
    download.
    """
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="ignore")


def load_tokenizer(name_or_path: str | None = None) -> Tokenizer:
    """Load a real Hugging Face tokenizer if `name_or_path` is given
    (requires Hub access for standard model names), otherwise return the
    offline `ByteTokenizer` fallback.
    """
    if name_or_path is None:
        return ByteTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "load_tokenizer(name_or_path=...) requires the `transformers` "
            "package (pip install transformers)."
        ) from e
    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load tokenizer '{name_or_path}' via "
            "AutoTokenizer.from_pretrained -- this typically requires "
            "network access to huggingface.co, which may be unavailable. "
            "Pass name_or_path=None to use the offline ByteTokenizer "
            "fallback instead, or supply a locally cached tokenizer."
        ) from e


# ---------------------------------------------------------------------------
# Text sources
# ---------------------------------------------------------------------------

# A small (~1.1MB) public-domain text corpus hosted on GitHub, reachable
# even when huggingface.co is not. This is NOT WikiText2 -- it's provided
# only as a stand-in with realistic text structure for offline testing.
_DEMO_CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_text_corpus(dest: str | Path, url: str = _DEMO_CORPUS_URL) -> Path:
    """Download a text corpus from an allowlisted URL to `dest` (skips the
    download if `dest` already exists). Returns the local path.

    Defaults to a small public-domain corpus reachable over GitHub's raw
    content CDN; pass a different `url` to fetch something else (must be
    on an allowlisted domain in this sandbox: github.com,
    raw.githubusercontent.com, codeload.githubusercontent.com).
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        data = resp.read()
    dest.write_bytes(data)
    return dest


def load_wikitext2_raw(split: str = "train", local_path: str | Path | None = None) -> str:
    """Load WikiText2's raw text (the paper's actual calibration source).

    Args:
        split: "train"/"validation"/"test" -- only used when fetching via
            the `datasets` library.
        local_path: path to a local raw WikiText2 text file (e.g. already
            downloaded `wiki.train.raw`). If given, this is used directly
            and no network access is attempted.

    Returns:
        The raw text as a single string.

    Raises:
        RuntimeError: if `local_path` is not given and the dataset cannot
            be fetched (e.g. no `datasets` package, or no Hub network
            access -- both expected in this sandbox).
    """
    if local_path is not None:
        return Path(local_path).read_text(encoding="utf-8", errors="ignore")

    try:
        import datasets
    except ImportError as e:
        raise RuntimeError(
            "load_wikitext2_raw() with no local_path requires the "
            "`datasets` package and network access to huggingface.co to "
            "fetch 'wikitext'/'wikitext-2-raw-v1'. Neither is guaranteed "
            "available in this environment. Either `pip install datasets` "
            "and ensure Hub access, or download the WikiText2 raw files "
            "yourself and pass their path as `local_path`."
        ) from e

    try:
        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to fetch WikiText2 via `datasets.load_dataset` -- this "
            "requires network access to huggingface.co. Download the raw "
            "WikiText2 files yourself and pass their path as `local_path`."
        ) from e
    return "\n".join(ds["text"])


# ---------------------------------------------------------------------------
# Segment sampling
# ---------------------------------------------------------------------------

def build_calibration_set(
    source: str | list[int] | torch.Tensor,
    tokenizer: Tokenizer | None,
    config: CalibrationConfig = CalibrationConfig(),
) -> torch.Tensor:
    """Sample `config.num_samples` random contiguous segments of
    `config.seq_len` tokens each, matching the paper's calibration
    protocol (128 random 2048-token segments from WikiText2).

    Args:
        source: either raw text (str, tokenized via `tokenizer`), a
            pre-tokenized list[int] of token ids, or a 1-D LongTensor of
            token ids.
        tokenizer: required if `source` is a raw string; ignored
            otherwise.
        config: sampling configuration (num_samples, seq_len, seed).

    Returns:
        LongTensor of shape (num_samples, seq_len).

    Raises:
        ValueError: if the source has fewer tokens than `seq_len`
            (can't extract even one full segment).
    """
    if isinstance(source, str):
        if tokenizer is None:
            raise ValueError("tokenizer is required when source is raw text")
        ids = tokenizer.encode(source)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
    elif isinstance(source, torch.Tensor):
        ids_tensor = source.to(torch.long).flatten()
    else:  # list[int]
        ids_tensor = torch.tensor(source, dtype=torch.long)

    total = ids_tensor.shape[0]
    if total < config.seq_len:
        raise ValueError(
            f"source has only {total} tokens, need at least "
            f"config.seq_len={config.seq_len} to extract one segment"
        )

    # Random start offsets, sampled with replacement across the valid
    # range -- matches the paper's "128 randomly selected 2048 token
    # segments" (segments may overlap; this is standard for GPTQ/AWQ/
    # OmniQuant-style calibration sampling, not a disjoint partition).
    rng = random.Random(config.seed)
    max_start = total - config.seq_len
    starts = [rng.randint(0, max_start) for _ in range(config.num_samples)]

    segments = torch.stack([ids_tensor[s: s + config.seq_len] for s in starts], dim=0)
    return segments


def make_synthetic_calibration_set(
    vocab_size: int,
    config: CalibrationConfig = CalibrationConfig(),
) -> torch.Tensor:
    """Fully synthetic calibration data (uniform-random token ids) -- NOT
    representative of real language statistics, provided for pipeline
    testing when no text/tokenizer is available at all.

    Returns:
        LongTensor of shape (num_samples, seq_len), values in
        [0, vocab_size).
    """
    g = torch.Generator()
    g.manual_seed(config.seed)
    return torch.randint(0, vocab_size, (config.num_samples, config.seq_len), generator=g)


def iterate_calibration_batches(calib_ids: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    """Yield `calib_ids` in contiguous chunks of `batch_size` rows (the
    last chunk may be smaller). Order is preserved -- callers wanting
    shuffled batches should shuffle `calib_ids`'s rows beforehand.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    n = calib_ids.shape[0]
    for start in range(0, n, batch_size):
        yield calib_ids[start: start + batch_size]