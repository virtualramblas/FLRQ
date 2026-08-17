from transformers import OPTConfig, OPTForCausalLM
from data.calib import download_text_corpus, ByteTokenizer, build_calibration_set, CalibrationConfig
from runner.opt_adapter import quantize_opt_model
from runner.eval_ppl import evaluate_model_ppl

cfg = OPTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 ffn_dim=64, vocab_size=256, max_position_embeddings=512,
                 word_embed_proj_dim=32)
model = OPTForCausalLM(cfg).eval()

text = download_text_corpus(".wikitext").read_text()  # real text, not WikiText2
tokenizer = ByteTokenizer()  # offline stand-in, not OPT's real BPE tokenizer

calib_ids = build_calibration_set(text, tokenizer, CalibrationConfig(num_samples=4, seq_len=128))
ppl_before = evaluate_model_ppl(model, text[:20000], tokenizer, seq_len=128)

result = quantize_opt_model(model, calib_ids, bits=4, x=0.3, epochs=3)
ppl_after = evaluate_model_ppl(model, text[:20000], tokenizer, seq_len=128)

print(ppl_before.ppl, "->", ppl_after.ppl)
# PPL before quantization: 256.256 over 156 windows
# 12 layers quantized, avg rank=0.1, avg extra-bit fraction=0.016
# PPL after quantization:  256.659 over 156 windows