"""
Unit tests for data/calib.py.

Kept fully hermetic (no network calls) so the suite is fast and
repeatable: `download_text_corpus` / real `load_wikitext2_raw` fetches are
exercised manually (see the bottom of this file / the accompanying
summary), not in the automated test list, since network availability
shouldn't gate CI-style runs. The informative-error path of
`load_wikitext2_raw` *is* tested here, since `datasets` genuinely isn't
installed in this environment -- that's a real, deterministic code path.

Checks:
1. ByteTokenizer encode/decode round-trips ASCII and UTF-8 text.
2. build_calibration_set from raw text produces the right shape, values
   are valid byte ids, and results are reproducible given a seed.
3. Sampled segments are genuinely contiguous substrings of the source
   token sequence (verified via string search against the ByteTokenizer's
   underlying byte source) -- a real end-to-end sanity check that "random
   2048-token segments" actually get extracted correctly.
4. build_calibration_set also accepts a pre-tokenized list[int] and a 1-D
   LongTensor directly (no tokenizer needed).
5. build_calibration_set raises a clear error when the source is shorter
   than seq_len.
6. Different seeds give different samples; same seed is reproducible.
7. make_synthetic_calibration_set: correct shape, values within
   [0, vocab_size), reproducible given a seed.
8. iterate_calibration_batches: correct chunk sizes (including a
   remainder), preserves order, covers every row exactly once.
9. load_wikitext2_raw with a local_path reads the file directly (no
   network needed).
10. load_wikitext2_raw with no local_path and no `datasets` installed
    raises a RuntimeError with a message pointing at the real fix
    (local_path / network access) -- exercises the actual fallback path
    live in this sandboxed environment.
11. load_tokenizer(None) returns a working ByteTokenizer.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.calib import (  # noqa: E402
    ByteTokenizer,
    CalibrationConfig,
    build_calibration_set,
    iterate_calibration_batches,
    load_tokenizer,
    load_wikitext2_raw,
    make_synthetic_calibration_set,
)


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "Hello, world! Résumé — café."
    ids = tok.encode(text)
    assert all(0 <= i < 256 for i in ids)
    assert tok.decode(ids) == text


def test_build_calibration_set_shape_and_values():
    tok = ByteTokenizer()
    text = "abcdefghij " * 500  # plenty of bytes
    cfg = CalibrationConfig(num_samples=8, seq_len=64, seed=0)
    calib = build_calibration_set(text, tok, cfg)
    assert calib.shape == (8, 64)
    assert calib.dtype == torch.long
    assert calib.min().item() >= 0
    assert calib.max().item() < 256


def test_sampled_segments_are_real_contiguous_substrings():
    tok = ByteTokenizer()
    text = "The quick brown fox jumps over the lazy dog. " * 50
    ids_full = tok.encode(text)
    cfg = CalibrationConfig(num_samples=5, seq_len=20, seed=1)
    calib = build_calibration_set(text, tok, cfg)

    ids_full_list = ids_full
    for row in calib:
        row_list = row.tolist()
        decoded_segment = tok.decode(row_list)
        assert decoded_segment in text, "sampled segment is not a real substring of the source text"
        # Also verify at the byte-id level directly (stronger than string
        # containment, avoids any decode-boundary edge cases).
        found = any(
            ids_full_list[i:i + len(row_list)] == row_list
            for i in range(len(ids_full_list) - len(row_list) + 1)
        )
        assert found


def test_build_calibration_set_accepts_pretokenized_list_and_tensor():
    cfg = CalibrationConfig(num_samples=4, seq_len=10, seed=2)
    ids_list = list(range(100))
    calib_from_list = build_calibration_set(ids_list, tokenizer=None, config=cfg)
    assert calib_from_list.shape == (4, 10)

    ids_tensor = torch.arange(100)
    calib_from_tensor = build_calibration_set(ids_tensor, tokenizer=None, config=cfg)
    assert calib_from_tensor.shape == (4, 10)
    # Same seed -> same sampled start offsets -> identical result.
    assert torch.equal(calib_from_list, calib_from_tensor)


def test_build_calibration_set_raises_on_too_short_source():
    cfg = CalibrationConfig(num_samples=2, seq_len=50, seed=0)
    short_ids = list(range(10))
    try:
        build_calibration_set(short_ids, tokenizer=None, config=cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_seed_controls_reproducibility_and_variation():
    tok = ByteTokenizer()
    text = "abcdefghijklmnopqrstuvwxyz " * 200
    cfg_a = CalibrationConfig(num_samples=6, seq_len=32, seed=42)
    cfg_a2 = CalibrationConfig(num_samples=6, seq_len=32, seed=42)
    cfg_b = CalibrationConfig(num_samples=6, seq_len=32, seed=43)

    calib_a = build_calibration_set(text, tok, cfg_a)
    calib_a2 = build_calibration_set(text, tok, cfg_a2)
    calib_b = build_calibration_set(text, tok, cfg_b)

    assert torch.equal(calib_a, calib_a2)
    assert not torch.equal(calib_a, calib_b)


def test_make_synthetic_calibration_set():
    cfg = CalibrationConfig(num_samples=5, seq_len=16, seed=7)
    calib = make_synthetic_calibration_set(vocab_size=50, config=cfg)
    assert calib.shape == (5, 16)
    assert calib.min().item() >= 0
    assert calib.max().item() < 50

    calib_again = make_synthetic_calibration_set(vocab_size=50, config=cfg)
    assert torch.equal(calib, calib_again)


def test_iterate_calibration_batches_covers_all_rows_in_order():
    calib = torch.arange(10 * 4).reshape(10, 4)
    batches = list(iterate_calibration_batches(calib, batch_size=3))
    sizes = [b.shape[0] for b in batches]
    assert sizes == [3, 3, 3, 1]
    reconstructed = torch.cat(batches, dim=0)
    assert torch.equal(reconstructed, calib)


def test_load_wikitext2_raw_with_local_path(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "wiki.train.raw"
        p.write_text("some wikitext-like content\nwith multiple lines\n", encoding="utf-8")
        text = load_wikitext2_raw(local_path=p)
        assert "wikitext-like content" in text


def test_load_wikitext2_raw_without_local_path_raises_informative_error():
    """`datasets` is genuinely not installed in this environment (verified
    separately), so this exercises the real fallback error path."""
    try:
        load_wikitext2_raw()
        raised = False
        msg = ""
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised
    assert "local_path" in msg or "network" in msg


def test_load_tokenizer_none_returns_byte_tokenizer():
    tok = load_tokenizer(None)
    assert isinstance(tok, ByteTokenizer)
    assert tok.encode("hi") == [104, 105]


if __name__ == "__main__":
    tests = [
        test_byte_tokenizer_roundtrip,
        test_build_calibration_set_shape_and_values,
        test_sampled_segments_are_real_contiguous_substrings,
        test_build_calibration_set_accepts_pretokenized_list_and_tensor,
        test_build_calibration_set_raises_on_too_short_source,
        test_seed_controls_reproducibility_and_variation,
        test_make_synthetic_calibration_set,
        test_iterate_calibration_batches_covers_all_rows_in_order,
        test_load_wikitext2_raw_with_local_path,
        test_load_wikitext2_raw_without_local_path_raises_informative_error,
        test_load_tokenizer_none_returns_byte_tokenizer,
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