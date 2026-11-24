# Model files and tokenizer

[Back to the README](../README.md)

## File formats

Models built in PyTorch with the [indranet package](../python/README.md) are exported in one file:

```python
from indranet import DecoderLM, ModelConfig
from indranet.export import export_model

model = DecoderLM.from_preset("attn_only_2l")   # or any config; see python/README.md
...                                             # train it
export_model(model, "model.safetensors")        # FP32 weights + config in one file
export_model(model, "model.safetensors", dtype="float16")  # weight matrices in fp16: half the file
export_model(model, "model.safetensors", dtype="bfloat16") # ... or in bf16
```

An fp16 file holds the same values `indra --fp16` would round a float32 file to; with `--fp16` it
loads them as they are. A bf16 file (also the usual format of published checkpoints) loads as
float32 exactly, or with `--fp16` converted to fp16: exactly, except values under about 7.6e-6 and
weights with values beyond fp16's range, which stay float32. bf16 has fp32's range but fewer
significant bits than fp16 (8 against 11).

The engine memory-maps model files. Some matrices are used in place, with no copy, the GPU
included: those stored in the type the engine keeps them in (F16 with `--fp16`, F32 otherwise) and
starting on a page boundary, as files from the Hugging Face converter do. Loading then reads
nothing ahead of time, and the weights take no memory of the process's own, since they're shared
with the system's file cache. Other matrices (bf16, or stored in the other type) are converted as
they're read, as before. So for a converted model, `--dtype float16` gives the fastest and leanest
`--fp16` runs.

## Converting Hugging Face checkpoints

Convert a pretrained Hugging Face checkpoint (Llama 1–3.x, Mistral, Qwen2/3, Phi-3/3.5/4-mini,
Gemma 1/2/3, GPT-2, Pythia) with the independent converter in [python/indranet_hf](../python/indranet_hf/README.md),
which needs only numpy:

```bash
python -m indranet_hf path/to/checkpoint model.safetensors    # or a Hub repo id
```

## Text input

Tokens are ids: `--tokens 5,17,3`, or `--repeat-random L` for a random segment repeated twice. Or
pass `--text "..."` (`--clean-text` / `--corrupt-text` for `patch`):

- **Converted Hugging Face models:** the text goes through the model's own tokenizer, which the
  converter stores in the file. The ids are those of Hugging Face's `tokenizers`, BOS included
  (leave it out with `--no-special-tokens`). `indra tokenize model --text "..."` shows them.
- **indranet models exported with a vocabulary** (`export_model(model, path, vocab=[...])`): each
  step takes the longest matching vocabulary entry.

`--target` and `--other` also accept a token's text, and all output shows tokens as text. Run
`build/indra --help` for every option.

The engine's tokenizer reads `tokenizer.json`, and implements what these models use:
- BPE with merges, byte fallback, `ignore_merges` and unknown tokens;
- the NFC, NFD, Replace and Prepend normalizers;
- the ByteLevel, Split (regex or string, every behavior), Metaspace and Digits pre-tokenizers;
- added and special tokens;
- the template, ByteLevel, Roberta and Bert post-processors.

Other parts (Unigram or WordPiece models, NFKC) are refused by name. The converter then stores only
the vocabulary, and `--text` falls back to longest match. Its Unicode data is version 13.0, so text
with characters assigned later may split differently from Hugging Face.

Decoding (`generate`'s output, `Model::decode`) is worked out from the same pipeline: byte-level
tokens become their bytes, SentencePiece-style tokens get their spaces and `<0xNN>` bytes back, and
added tokens are their text. This gives what Hugging Face's decoders give for these models.
