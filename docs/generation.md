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
occurrence of the last `--ngram` tokens (3 by default), continuing the repeat past the end of the text
when that occurrence is close (after `x y x y` it proposes `x y x y ...`). That costs nothing and pays
off where the output repeats the input or itself: editing, summarizing, quoting, code, loops. Steps propose fewer tokens after a
rejection. If proposals keep failing it stops proposing, and stops running the draft, for a few
tokens, then longer each time, so text it can't predict costs little. A draft with a smaller vocabulary
stops proposing once the text holds a token it has no id for, and the model continues alone.

The proposals only change the speed. Greedy output is the same as without speculation, up to
rounding: the checking pass computes several rows at once. Sampled output has the same distribution,
by the standard acceptance rule, though not the same tokens for a given seed.

```
build/indra generate llama.safetensors --text "$(cat prompt.txt)" --speculate 8
build/indra generate llama8b.safetensors --text "Once upon a time" --speculate 4 --draft llama1b.safetensors
```

### Hooks and speculation

Hooks don't need to know about speculation. With or without `speculate`, they see the same calls in
the same order:

```
the prompt's hooks → token 1 → token 1's hooks → token 2 → token 2's hooks → ...
```

The prompt arrives in one call. After that, each call holds one token's row, and a hook never runs on
a proposal that gets rejected. So a hook can use `a[-1]`, keep a count or a log, draw random numbers, or
use a value saved from an earlier token, exactly as it would without `speculate`.

While a pass computes proposals, their rows get a *prediction* of what the hook will do, learned from
its earlier calls: leaving the row alone, overwriting values, scaling or adding an offset, including
repeating patterns such as one edit on every second token. Once a proposal is confirmed, the real hook
runs on its recorded activation. If it does what was predicted, the batched row stands. If not, that
token is computed again from the hook's block with the real output: the blocks below it were right.
The proposals after it stay, since their lower blocks are right too, and each one that is confirmed is
computed again from that block. At `hook_final`, after the last block, only that token's logits are
computed again, from the hook's real output: nothing else reads it, so the proposals after it stand as
they are.

So a hook only affects the speed:

| Hook | With `speculate` |
|---|---|
| The same edit each token, or a repeating pattern of edits | Keeps most of the speedup |
| Read-only hooks and captures | Keeps the speedup |
| Edits that can't be predicted (random noise, a value from another token or layer), in the upper half of the blocks | Keeps part of the speedup: each token is computed again from the hook's block |
| Edits that can't be predicted, in the lower half | About the speed of ordinary decoding: proposals wait until predictions have been right |

A prediction that leaves a row alone or overwrites values must match the hook's output exactly, so an
edit of any size counts. A predicted scale or offset is compared up to rounding (a relative difference of
about 1e-6), since redoing one, such as adding a vector, can differ in the last bits.

As in ordinary decoding, where each pass runs a copy of the hooks, each token's hooks run on a copy made
just before they run: state a C++ hook keeps in itself (a `mutable` lambda's) lasts one token, and the
`Hooks` object is never changed. Filters are asked then too, so a filter that turns on or off between
tokens takes effect at the same token as without `speculate`.

From C++, `DecodeRuntime` (`tfi/runtime.h`) gives the same decoding one token at a time, and a token's hooks run when the
next one is asked for:

```cpp
DecodeRuntime runtime(m, prompt, options, &hooks);
while (std::optional<int64_t> token = runtime.next()) { /* ... */ }
```
