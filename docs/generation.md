# Text generation

[Back to the README](../README.md)

```
build/indra generate llama.safetensors --text "The capital of France is" --max-new 30
build/indra generate llama.safetensors --text "Once upon a time" --temperature 0.8 --top-p 0.9 --seed 1
```

A seed picks the same tokens on every platform, given the same logits: the sampler's weights and
draws use only exactly specified arithmetic, and it finds `--top-k` and `--top-p` in linear time
rather than by sorting the vocabulary.

## Trace and batches

`--trace` adds a table: for each new token, its probability and the alternatives, the first
logit-lens stage where it was the top prediction, and the largest parts of its logit (direct logit
attribution), all from the pass that chose it.

From C++ and Python, `generate_batch` continues several prompts at once, one stacked pass per step,
so the weights are read once for all of them. Each sequence comes out as `generate` would give it
(seed + i for sampling), and can have hooks of its own: for example the same prompt with and without
an intervention, side by side.

## KV cache

`generate` runs the prompt in one pass and then each new token alone, reading the earlier
positions' keys and values from a KV cache. On macOS, models of a few hundred million weights or
more generate on the GPU, with the cache kept there only: its CPU copy is filled when something reads
it (a CPU step, `keys()`, a copy), so a long context isn't held twice. With `--fp16` (fp16 weights)
the cache holds fp16 keys and values too: half its memory, rounded like the weights; float32 models
keep it exact. It stops at the model's end-of-text tokens (stored for
converted models; `--no-eos` ignores them), at `--stop` ids, or after `--max-new` tokens.

## Speculative decoding

`--speculate K` turns on speculative decoding. Each step proposes up to K tokens, and the model
checks them all in one pass, which costs about as much as one token since its weights are read once.
It keeps the tokens it agrees with, plus one of its own. The proposals come from `--draft FILE`, a
smaller model with the same tokenizer (Llama 3.2 1B for Llama 3.1 8B, Qwen2.5 0.5B for the larger
Qwen2.5 models). Without a draft they come from earlier in the text: the tokens that followed the last
occurrence of the last `--ngram` tokens (3 by default). That costs nothing and pays off where the
output repeats the input: editing, summarizing, quoting, code. Steps propose fewer tokens after a
rejection, so on text that doesn't repeat it costs almost nothing. A draft with a smaller vocabulary
stops proposing once the text holds a token it has no id for, and the model continues alone.

The proposals only change the speed. Greedy output is the same as without speculation, up to
rounding: the checking pass computes several rows at once. Sampled output has the same distribution,
by the standard acceptance rule, though not the same tokens for a given seed.

```
build/indra generate llama.safetensors --text "$(cat prompt.txt)" --speculate 8
build/indra generate llama8b.safetensors --text "Once upon a time" --speculate 4 --draft llama1b.safetensors
```

### Hooks and speculation

Hooks also run on proposals that are then rejected; those rows are dropped from the cache and the
trace. Because attention is causal, the kept rows are what they would be without speculation for
hooks that treat each row alike, such as editing a head or scaling every row. A hook that picks a row
by index (`a[-1]` is then the last proposal, not the current token), reads other rows or carries state
from one pass to the next (a count, a value from an earlier step) can change the output, so run those
without `speculate`. Saving all of an activation's rows at one layer to use at a later layer of the
same pass is fine.

For example, say the text ends in `C` and the proposals are `D E F`:

```python
def edit_last(a, name):
    a[-1] *= -1

hooks = {"hook_final": edit_last}
tfi.generate(m, t, hooks=hooks)               # a[-1] is each new token in turn
tfi.generate(m, t, hooks=hooks, speculate=4)  # a[-1] is the last proposal: the text can differ
```

```
without speculate:  [C]  [D]  [E]  [F]    each pass edits its own token
with speculate:     [C, D, E, F]          one pass: only F's row is edited
```

Row `C` decides whether `D` is kept, row `D` whether `E` is, and row `E` whether `F` is, so all three
checks use rows the hook never edited.
