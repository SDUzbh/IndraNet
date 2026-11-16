# indranet_hf: Hugging Face checkpoints for the engine

Converts a Hugging Face checkpoint into an IndraNet model file that `indra` and the C++ API load. It
is independent of the `indranet` package and of PyTorch: it needs only numpy, plus `huggingface_hub`
if you want it to download from the Hub.

```bash
python -m indranet_hf path/to/checkpoint model.safetensors           # a folder with config.json + *.safetensors
python -m indranet_hf Qwen/Qwen2.5-0.5B qwen.safetensors             # a Hub repo id (pip install huggingface_hub)
python -m indranet_hf path/to/checkpoint model.safetensors --dtype float16
python -m indranet_hf path/to/checkpoint --config-only               # print the IndraNet config, write nothing
build/indra info model.safetensors
```

Run it from the repository root with `python/` on the path (`pip install -e python` does that), or:

```python
from indranet_hf import convert, translate_config
summary = convert("path/to/checkpoint", "model.safetensors", dtype="auto")
```

## Supported models

| `model_type` | Models | Notes |
|---|---|---|
| `llama` | Llama 1/2/3.x, TinyLlama, SmolLM, ... | optional `attention_bias` / `mlp_bias` |
| `mistral` | Mistral 7B v0.x | sliding window |
| `qwen2` | Qwen1.5, Qwen2, Qwen2.5 | q/k/v biases; sliding layers above `max_window_layers` |
| `qwen3` | Qwen3 (dense) | per-head q/k RMSNorm |
| `phi3` | Phi-3, Phi-3.5, Phi-4-mini | fused qkv and gate/up split; partial RoPE |
| `gemma` | Gemma 1 | `1 + w` norms, scaled embeddings, GeGLU |
| `gemma2` | Gemma 2 | plus sandwich norms, softcaps, alternating sliding layers |
| `gemma3_text` | Gemma 3 text models | plus q/k norms, local RoPE base on sliding layers, scaling on global ones |
| `gpt2` | GPT-2 | Conv1D weights transposed, fused qkv split |
| `gpt_neox` | Pythia, GPT-NeoX | per-head fused qkv, partial RoPE, parallel residual |

Refused, with a message saying why: mixture-of-experts and other model types (among them Gemma 3's
multimodal checkpoints, `model_type` `gemma3`, which the 4B and larger models ship as), and any
tensor the converter doesn't recognise (rather than silently dropping a weight). Weights must be
safetensors: PyTorch pickles (`.bin`) are not read, because loading one can run code.

RoPE scaling (`rope_scaling` / `rope_parameters`) carries over: `linear`, `dynamic`, `llama3`
(Llama 3.1+), `yarn` and `longrope` (Phi-3.5, Phi-4-mini, the 128K Phi-3 models). `dynamic` and
`longrope` change their frequencies past `original_max_position_embeddings` (4096 for Phi-3.5);
the engine computes them exactly up to that length and refuses longer sequences.

## What it writes

- **Config:** IndraNet's own config, with every field explicit. Hugging Face's `config.json` alone
  doesn't describe the computation: part of it is in each family's modeling code, and that part is
  encoded here (`families.py`). Layers that differ (sliding window, RoPE base) become
  `layer_overrides`. The original `config.json` is kept in the file's metadata as `hf_config`.
- **Weights:** renamed to IndraNet's names, with fused matrices split and transposed ones turned
  round. Weight matrices keep the checkpoint's type by default (bf16 stays bf16), or take
  `--dtype float32|float16|bfloat16`; a matrix with a value outside the asked type's range stays
  float32, with a warning. Norm scales and biases are float32 (Gemma's `1 + w` is folded in
  there). A tied output matrix is stored once. Only one tensor is in memory at a time.
- **Biases:** the config's `bias` names exactly the parts that have them: `true`, `false`, or a list
  such as Qwen2's `["qkv"]` (parts: `qkv`, `o`, `mlp`, `norm`). The engine requires those biases and
  refuses any other. A part that is only partly there (a q bias without k and v) is refused.
- **Vocabulary:** the tokenizer's token strings (from `tokenizer.json`, or GPT-2's `vocab.json`),
  decoded for display, so results print tokens as text. This is not the model's tokenizer: `indra
  --text` matches the longest token at each point, which can split text differently from BPE. For
  exact inputs, tokenize with the model's tokenizer and pass ids:

  ```python
  from transformers import AutoTokenizer
  ids = AutoTokenizer.from_pretrained("path/to/checkpoint")("The capital of France is")["input_ids"]
  print(",".join(map(str, ids)))   # then: build/indra ... --tokens <ids>
  ```

## Tests

`python/tests/test_hf.py`: unit tests (config translation for every family, older and newer config
keys, refusals, bf16 rounding against torch, vocabulary decoding), and golden tests that need
`transformers` and a C++ toolchain. Those build a tiny random model of each family with
transformers, save it in float32 and in bf16, convert it, and check the engine reproduces
transformers' logits and every layer's residual stream to 1e-4 relative, loading the bf16 files both
as float32 and with `--fp16`.
