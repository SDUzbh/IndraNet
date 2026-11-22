# IndraNet

[![CI](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml/badge.svg)](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml)

IndraNet takes trained decoder-only transformers apart. Its core is **tfi**, a dependency-free
C++17 engine that re-implements the forward and backward pass with named hook points and runs the
standard reverse-engineering methods on top: the logit lens, direct logit attribution, head scores,
activation patching (exact, attribution and mixed), ablation, and weight-only circuit analysis. It
also generates text with a KV cache, with hooks applied at every step, to see what an intervention
does to what the model writes.

The Python package [`indranet`](python/) supplies the models to study. It builds and trains them in
PyTorch from a single config, with the same hook names as the engine, and exports them in one file.
[`indranet_hf`](python/indranet_hf/) converts Hugging Face checkpoints, and [`tfi`](python/tfi/) gives
Python the engine itself (numpy in and out).

```
indranet (PyTorch)  --train-->  model  --export_model-->  model.safetensors  -->  indra (C++, tfi engine)
Hugging Face checkpoint  --indranet_hf-->  model.safetensors  -->  tfi (Python bindings)
```

| Path | What it is |
|---|---|
| [include/tfi/](include/tfi/), [src/](src/) | The tfi engine: `build/libtfi.a`, namespace `tfi` |
| [tools/indra.cpp](tools/indra.cpp) | The `indra` command line |
| [tests/](tests/) | C++ unit tests and the golden-test runner |
| [python/](python/) | The `indranet` package: model building, training helpers, export, its tests |
| [python/indranet_hf/](python/indranet_hf/) | The Hugging Face converter |
| [python/tfi/](python/tfi/) | Python bindings for the engine (nanobind), built with `make python` |
| [examples/](examples/) | Two circuits trained in Python and reverse-engineered with `indra` |

## Build

```bash
make            # build/indra (CLI), build/libtfi.a, build/tfi_tests, build/tfi_golden
make test       # C++ unit tests
make python     # the Python bindings, python/tfi/_tfi*.so (needs CMake and `pip install nanobind`)
```

Or with CMake: `cmake -S . -B build-cmake && cmake --build build-cmake`. On macOS, matrix
multiplies use Accelerate and the GPU; elsewhere a portable loop is used. `indra --device` picks
where work runs: `both` (the default) sends work to the GPU when it is large enough to repay it,
`cpu` never uses the GPU (also `--no-gpu`), and `gpu` runs everything it can on the GPU, attention
included. macOS builds need macOS 15 or newer.

## Get a model

```bash
pip install -e "python[dev]"
```

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

Or train an example circuit and export it in one go:

```bash
python examples/induction_heads.py --export induction.safetensors
```

Or convert a pretrained Hugging Face checkpoint (Llama 1–3.x, Mistral, Qwen2/3, Phi-3/3.5/4-mini,
Gemma 1/2/3, GPT-2, Pythia) with the independent converter in [python/indranet_hf](python/indranet_hf/README.md),
which needs only numpy:

```bash
python -m indranet_hf path/to/checkpoint model.safetensors    # or a Hub repo id
```

## Commands

| Command | Kind | What it answers |
|---|---|---|
| `info` | – | Architecture, parameter count, hook point names |
| `run` | activations | Next-token predictions and loss |
| `lens` | activations | Logit lens: what the residual stream predicts after each sublayer |
| `dla` | activations | Direct logit attribution: which heads/MLPs write the answer directly |
| `heads` | activations | Per-head pattern scores (previous-token, duplicate, induction, first-token, self) and two-hop induction pairs at any lag |
| `patch` | activations | Activation patching clean → corrupted, per layer × position or per head (normalized, or raw with `--raw`). `--method attribution` estimates every site from gradients; `--method mixed` then measures the top estimates exactly |
| `ablate` | activations | Loss increase when each head is zero-ablated, or mean-ablated with means from reference sequences |
| `circuits` | weights | Q/K/V composition scores vs. a random baseline, single-head and two-head OV copying |
| `attn` | activations | Where each head of a layer attends, query by query (top keys with weights) |
| `tokenize` | – | Token ids for text, as the model's tokenizer gives them (reads no weights) |
| `generate` | – | Continues the input, printing the text as it comes: greedy, or sampled with `--temperature`, `--top-k`, `--top-p` and `--seed`; `--speculate K` for speculative decoding |

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

```
build/indra generate llama.safetensors --text "The capital of France is" --max-new 30
build/indra generate llama.safetensors --text "Once upon a time" --temperature 0.8 --top-p 0.9 --seed 1
```

`--trace` adds a table: for each new token, its probability and the alternatives, the first
logit-lens stage where it was the top prediction, and the largest parts of its logit (direct logit
attribution), all from the pass that chose it.

From C++ and Python, `generate_batch` continues several prompts at once, one stacked pass per step,
so the weights are read once for all of them. Each sequence comes out as `generate` would give it
(seed + i for sampling), and can have hooks of its own: for example the same prompt with and without
an intervention, side by side.

`generate` runs the prompt in one pass and then each new token alone, reading the earlier
positions' keys and values from a KV cache. On macOS, models of a few hundred million weights or
more generate on the GPU, with the cache kept there only: its CPU copy is filled when something reads
it (a CPU step, `keys()`, a copy), so a long context isn't held twice. With `--fp16` (fp16 weights)
the cache holds fp16 keys and values too: half its memory, rounded like the weights; float32 models
keep it exact. It stops at the model's end-of-text tokens (stored for
converted models; `--no-eos` ignores them), at `--stop` ids, or after `--max-new` tokens.

`--speculate K` turns on speculative decoding. Each step proposes up to K tokens, and the model
checks them all in one pass, which costs about as much as one token since its weights are read once.
It keeps the tokens it agrees with, plus one of its own. The proposals come from `--draft FILE`, a
smaller model with the same tokenizer (Llama 3.2 1B for Llama 3.1 8B, Qwen2.5 0.5B for the larger
Qwen2.5 models). Without a draft they come from earlier in the text: the tokens that followed the last
occurrence of the last `--ngram` tokens (3 by default). That costs nothing and pays off where the
output repeats the input: editing, summarizing, quoting, code. Steps propose fewer tokens after a
rejection, so on text that doesn't repeat it costs almost nothing.

The proposals only change the speed. Greedy output is the same as without speculation, up to
rounding: the checking pass computes several rows at once. Sampled output has the same distribution,
by the standard acceptance rule, though not the same tokens for a given seed. Hooks also run on
proposals that are then rejected; those rows are dropped from the cache and the trace, and because
attention is causal the kept rows are what they would be without speculation.

```
build/indra generate llama.safetensors --text "$(cat prompt.txt)" --speculate 8
build/indra generate llama8b.safetensors --text "Once upon a time" --speculate 4 --draft llama1b.safetensors
```

## Walkthrough: the 2-layer induction circuit

On the model from `examples/induction_heads.py` (2 layers × 4 heads, attention-only, RoPE). The
transcripts below are from a real run, trimmed.

**Which heads form the circuit?** The textbook scores (attention to t−1, to the token after the
earlier copy) stay near zero. The lag-independent two-hop score finds the circuit anyway:

```
$ indra heads induction.safetensors --seg 20 --samples 16
previous token (t-1):
      L0  0.001  0.010  0.001  0.003
two-hop induction (reader attends to r, writer at r attends to the token to copy; any lag):
  L0H2 -> L1H3   0.920
  L0H0 -> L1H3   0.916
  L0H2 -> L1H1   0.914
```

**Where does the answer appear?** Layer 1 does the induction step:

```
$ indra lens induction.safetensors --repeat-random 20 --seed 1 --pos 30
stage     p(target)  rank
embed     0.0000     114
L0.attn   0.0017     112
L1.attn   0.4076     0
```

**Who writes it?** The layer-1 heads write it directly. The parts sum exactly to the real logit difference:

```
$ indra dla induction.safetensors --repeat-random 20 --seed 1 --pos 31 --target 101 --other 99
actual 7.2142   sum of parts 7.2142   (final norm scale frozen)
L1H3                2.4083  +########################################
L1H0                2.2056  +####################################
L1H1                1.8915  +###############################
```

**How does the information move?** Corrupt the source token (position 12) and patch activations back in:

```
$ indra patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 --site resid_pre
              ...     12  ...     21  ...
      L0       0  1.000    0       0
      L1       0      0    0   1.000
$ indra patch ... --site heads
             H0     H1     H2     H3
      L0  0.475 -0.009  0.549  0.001
```

The source token's information sits at position 12 before layer 0 and at position 21 = 12 + 9
before layer 1. The layer-0 heads H0 and H2 carry it with a lag of 9 (0.48 + 0.55 ≈ 1).

**Can the weights alone confirm it?** No single head copies tokens, but writer→reader OV paths do:

```
$ indra circuits induction.safetensors
OV copying: fraction of tokens a head's direct path maps back to themselves (chance 0.0078)
      L0  0.000  0.000  0.008  0.000
      L1  0.000  0.000  0.000  0.000
two-head OV path copying (writer OV, then reader OV):
  L0H2 -> L1H0   0.969
  L0H0 -> L1H0   0.953
```

Five methods, two of them weight-only, agree on one mechanism: lag-9 heads (L0H0, L0H2) shift each
token's information forward, and layer-1 heads match on it and copy the answer.

## Attribution patching

`indra patch` uses exact patching by default (`--method exact`): every number is measured. It runs
one forward pass per site, so its cost grows with layers × positions. Two optional methods trade
accuracy for speed. Attribution patching (Nanda 2023; Syed et al. 2023) estimates every site at once from one clean run,
one corrupted run and one backward pass:

```
effect ≈ (clean activation − corrupted activation) · ∂metric/∂activation   (gradient on the corrupted run)
```

The estimate is linear, so it is exact only where the logits depend linearly on a site. It
underestimates large changes that saturate attention or an MLP. `--steps N` averages the gradient
over N runs whose token embeddings move from the corrupted input towards the clean one. This is the
integrated-gradients variant of EAP-IG (Hanna et al. 2024), and it costs one more forward + backward
pass per step. `--compare` also runs exact patching and reports how well the two agree:

```
$ indra patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 \
      --site heads --method attribution --steps 8 --compare
             H0     H1     H2     H3
      L0  0.335 -0.008  0.491 -0.002
      L1  0.219  0.260  0.128  0.258
...
agreement over 8 sites: correlation 0.980, largest error 0.140 at L0 H0 (estimate 0.335, exact 0.475)
```

Agreement with exact patching on the two example models:

| Model, site | Plain (1 step) | `--steps 8` |
|---|---|---|
| Induction, `resid_pre` (80 sites) | r = 0.989, worst error 0.64 | r = 0.994, worst error 0.18 |
| Induction, heads (8) | r = 0.737, worst error 0.42 | r = 0.980, worst error 0.14 |
| Arithmetic carry, `resid_pre` (56 sites) | r = 0.971, worst error 0.89 | r = 0.977, worst error 0.47 |
| Arithmetic carry, heads (32) | r = 0.943, worst error 0.86 | r = 0.982, worst error 0.43 |

The estimates find the right sites, but their sizes are only rough. On the arithmetic carry, plain
attribution gives the carry head L0H1 0.13 where exact patching gives 0.99. With `--steps 8` it
gives 0.56 and ranks L0H1 first.

`--method mixed` combines the two methods. Attribution estimates every site, then exact patching
re-measures the `--verify K` sites with the largest estimates (K = 10 unless given). The measured
sites are exact; every other entry is still an estimate. It costs attribution plus K forward
passes. Entries marked `*` were measured exactly:

```
$ indra patch arith.safetensors --clean-text "58291+76834=+5" --corrupt-text "58297+76834=+1" \
      --pos 13 --target 2 --other 3 --site heads --method mixed --verify 5
* = measured by exact patching (the 5 largest estimates); other entries are attribution estimates

             H0      H1      H2      H3      H4      H5      H6      H7
      L0 -0.001   0.989*  0.132* -0.000   0.001   0.000   0.003  -0.004
      L1  0.003  -0.000  -0.007  -0.001   0.021* -0.001  -0.004   0.016*
      L2 -0.001  -0.002  -0.000   0.000  -0.005   0.003   0.013* -0.008
      L3  0.000   0.002   0.011  -0.002   0.001   0.001   0.002   0.001

the exactly measured sites' estimates were off by up to 0.855 (L0 H1: estimate 0.134, exact 0.989)
largest estimate left unverified: 0.011 (L3 H2)
```

The carry head's estimate was badly low, but it still ranked first, so it was measured exactly.
Against full exact patching, this table has a correlation of 0.999 and a largest error of 0.004.

The risk is a site whose estimate is so low that it falls below the cut. With plain attribution
and K = 3–5, that happens to induction head L0H0, estimated at 0.05 but really 0.48. `--steps 8`
ranks sites better and fixes it. Largest error against full exact patching:

| Model, site | K = 3 | K = 5 | K = 3, `--steps 8` | K = 10, `--steps 8` |
|---|---|---|---|---|
| Induction, `resid_pre` (80 sites) | 0.000 | 0.000 | 0.000 | 0.000 |
| Induction, heads (8) | 0.421 | 0.421 | 0.017 | 0.000 |
| Arithmetic carry, `resid_pre` (56) | 0.893 | 0.009 | 0.471 | 0.001 |
| Arithmetic carry, heads (32) | 0.008 | 0.004 | 0.028 | 0.012 |
| Arithmetic carry chain, `resid_pre` (60) | 0.961 | 0.006 | 0.398 | 0.014 |
| Arithmetic carry chain, heads (32) | 0.005 | 0.005 | 0.039 | 0.004 |

The output reports how far off the measured sites' estimates were and the largest estimate left
unmeasured. When the estimates were far off, unmeasured sites may be too. Exact patching (the
default) has no such risk.

The gradients come from `Model::run_with_gradients` (or `metric_gradients`): a backward pass that
produces activation gradients only, no weight gradients. It costs about one forward pass.

## C++ API

```cpp
#include "tfi/analysis.h"
#include "tfi/generate.h"
using namespace tfi;

Model m = Model::load("model.safetensors");                 // or (..., WeightFormat::fp16): half the memory
Tokens t = {5, 17, 3, 5, 17};

ActivationCache cache;                                       // every hook point, IndraNet names
Tensor logits = m.run_with_cache(t, cache);
const Tensor& pattern = cache.at("blocks.1.attn.hook_pattern");   // (H, T, T)

Hooks hooks;                                                 // interventions
hooks.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) { /* edit (T, H, D) in place */ });
Tensor edited = m.forward(t, &hooks);

DLAResult dla = direct_logit_attribution(m, t, /*pos=*/4, /*target=*/3, /*other=*/9);
Composition comp = composition_scores(m);                    // weights only

Tokens clean = t, corrupt = {5, 17, 8, 5, 17};
ActivationCache grads;                                       // d metric / d activation, every hook point
metric_gradients(m, t, /*pos=*/4, /*target=*/3, /*other=*/9, grads);
PatchResult atp = attribution_patch_heads(m, clean, corrupt, /*pos=*/4, /*target=*/3, /*other=*/9, /*steps=*/8);
PatchResult mix = mixed_patch_heads(m, clean, corrupt, 4, 3, 9, /*verify=*/10, /*steps=*/8);  // mix.exact flags measured entries

KVCache kv;                                                  // keys and values of the positions so far
Tensor first = m.forward_cached(t, kv);                      // = forward(t); kv now holds 5 positions
Tensor next = m.forward_cached({8}, kv);                     // only the new row: = forward(t + {8})'s last

GenerateOptions opt;                                         // tfi/generate.h
opt.max_new_tokens = 20;
opt.temperature = 0.8;                                       // 0 (default): greedy
Tokens more = generate(m, t, opt, &hooks);                   // hooks apply at every step
std::vector<TraceStep> trace;                                // per new token: prob, logit lens, DLA, attention
generate(m, t, opt, nullptr, &trace);                        // (opt.trace says which)
std::vector<Tokens> both = generate_batch(m, {t, t}, opt, {nullptr, &hooks});  // baseline and intervention, one pass a step
opt.speculative.draft_tokens = 8;                            // speculative decoding: prompt lookup,
opt.speculative.draft = &small;                              // or a draft model with the same tokenizer
Tokens fast = generate(m, t, opt);
std::string text = m.decode(more);

const Weight& Wq = m.weight("blocks.0.attn.q_proj.weight");  // float32 or fp16
Tensor q_rows = Wq.rows(0, 16);                              // read as float32 either way (also cols, to_tensor)
```

`Model::weight` returns a `Weight`. Matrices (float32 or fp16) are read with `rows()`, `cols()`,
`to_tensor()` and the `linear()` / `matmul()` overloads, which give float32 either way; they sit in
page-aligned memory that the GPU uses in place, so no weight is held twice. Norm scales and biases
are float32 tensors, read with `f32()`.

Link against `build/libtfi.a` with `-Iinclude` (on macOS also `-framework Accelerate -framework Metal
-framework MetalPerformanceShaders -framework Foundation`).

## Python bindings

`make python` builds the module `tfi` (with CMake and [nanobind](https://github.com/wjakob/nanobind))
into [python/tfi/](python/tfi/). It is used from the repository: `pytest` puts `python/` on the
path; elsewhere, set `PYTHONPATH=python`.

```python
import tfi

m = tfi.Model.load("model.safetensors", fp16=True)
t = m.tokenize("When Mary and John went to the store, John gave a drink to")
logits, cache = m.run_with_cache(t, names=["blocks.9.attn.hook_pattern"])   # numpy float32, (H, T, T)

def knock_out(z, name):                    # a hook gets a copy of the activation; editing it in place,
    z[:, 7, :] = 0                         # or returning a new array, replaces the activation
logits = m.forward(t, hooks={"blocks.15.attn.hook_z": knock_out})

mary, john = m.tokenize(" Mary", False)[0], m.tokenize(" John", False)[0]
dla = tfi.direct_logit_attribution(m, t, pos=-1, target=mary, other=john)     # labels, values, sum, actual
lens = tfi.lens_grid(m, t)                 # the logit lens at every position, one forward pass
r = tfi.patch(m, clean, corrupt, "resid_pre", pos=-1, target=mary, other=john, method="mixed")

new = tfi.generate(m, "The capital of France is", max_new_tokens=20)          # token ids, KV cache
m.decode(new)
tfi.generate(m, t, temperature=0.8, top_p=0.9, seed=1, hooks={"blocks.15.attn.hook_z": knock_out})
new, steps = tfi.generate(m, t, max_new_tokens=20, trace=True)   # per token: prob, top tokens, logit lens,
steps[0]["lens"]["rank"], steps[0]["dla"]["labels"]                # DLA (+ trace_attention=k: where heads looked)
keep = lambda a, name: print(name, a.shape)                        # captures only read: a GPU pass doesn't stop
tfi.generate(m, t, max_new_tokens=5, captures={"blocks.10.hook_resid_post": keep})
base, edited = tfi.generate_batch(m, [t, t], max_new_tokens=20,     # several at once; hooks per sequence
                                  hooks=[None, {"blocks.15.attn.hook_z": knock_out}])
stats = {}
tfi.generate(m, t, max_new_tokens=200, speculate=8, stats=stats)   # speculative (draft=small for a model)
cache = tfi.KVCache()                      # step by step: forward_cached runs only the new tokens
logits = m.forward_cached(t, cache)
logits = m.forward_cached([mary], cache)   # (1, vocab); hooks see the new rows
```

The functions mirror the C++ API (`logit_lens`, `head_scores`, `patch`, `metric_gradients`,
`head_ablation`, `composition_scores`, `ov_copying_scores`, `head_weights`, `generate`, `set_device`, ...).
Activations come back as numpy arrays that take over the engine's buffers, with no copy, and
analysis results as dicts. Every engine call releases the GIL, so threads can run it side by side.

## Correctness

- **Golden tests** (`python/tests/test_golden.py`) export ~130 model configurations and
  compare the C++ forward pass with PyTorch on the logits and every hook point. The configurations
  cover every norm × placement × positional encoding × MLP combination, GQA/MQA, sliding window,
  softcaps, parallel residual, every RoPE scaling type, per-layer overrides and all presets.
  Agreement is at float32 rounding level, about 3e-7 relative. The same configurations check the backward pass: the
  gradient of a metric at every hook point must match PyTorch autograd's to 1e-4.
- **Hugging Face golden tests** (`python/tests/test_hf.py`, with `transformers` installed) build a
  tiny random model of each supported family, convert it, and check the engine reproduces
  transformers' logits and every layer's residual stream, from float32 and bf16 checkpoints.
- **Tokenizer tests** (also `test_hf.py`): tokenizers of every kind the engine implements, trained
  by Hugging Face's `tokenizers` library, must give its ids on varied text (scripts, emoji,
  whitespace runs, decomposed accents, special tokens), with and without special tokens. The C++
  unit tests check UTF-8, NFC / NFD, the split-pattern regex and the BPE rules on their own.
- **Unit tests** (`tests/unit.cpp`) check invariants on random models: DLA parts sum to the real
  logit difference, the last logit-lens stage equals the model's output, patching the only
  changed input restores the clean run exactly, the fast low-rank composition formula equals the
  direct d×d computation, plus causality and hooks. Gradients at every hook point are checked
  against finite differences. Attribution patching must equal exact patching where the model is
  linear in a site. Mixed patching must measure exactly the top-K estimated sites and agree with
  exact patching (bit for bit when it measures every site). Batched patching must agree with
  naive patching for every batch size (on macOS with the GPU on and off), the GPU linear
  layer with the CPU one, and a fused GPU submission with separate calls (bit for bit). Batched
  ablation and reference means must agree with one plain pass per head and per reference. On
  macOS, each GPU kernel must match its CPU counterpart. Passes that keep the residual stream on
  the GPU must match both the plain forward pass and the one-multiply-at-a-time GPU path. That's
  checked for 14 model types, including post-norm and every activation, with hooks on both sides.
  On the same 14 types, batched GPU gradients must match one CPU backward pass per sequence at
  every hook point they report. Attribution and mixed patching on the GPU must match the CPU for
  any batch size and keep the CPU tests' invariants. Attention on the GPU (both kernels) and its
  backward pass must match the CPU's, also on sequences longer than the kernels' tiles, with head
  sizes from 4 to 160. In GPU-only mode a single pass must match the CPU at every activation it
  caches, with hooks that edit every activation (the pattern too) and capture hooks among them.
- **fp16 weights** (also `tests/unit.cpp`): the fp16 conversion must round trip every fp16 value,
  round to nearest-even and match the hardware's on arm64. A model with fp16 weights must compute
  what the float32 model with the rounded weights computes (forward and gradients), and so must
  loading with conversion. On the GPU, its batched passes and gradients must equal that float32
  model's bit for bit. The patching, attribution, DLA, lens and circuit invariants above are
  rerun on every configuration with fp16 weights.

## Approximations and limits

- **DLA** freezes the final norm's scale (standard practice). It is not defined for post-norm
  models. With sandwich norm it attributes whole attention layers, since per-head outputs pass
  through a norm.
- **Weight-based tools** fold each norm's weight into the adjacent matrices but ignore its scale
  and centering. Q/K composition ignores RoPE's position-dependent rotation (the tool prints a
  note). Compare composition scores with the printed random baseline, not against zero.
- **Head scores** use random repeated sequences, as is standard for finding induction heads.
- **Patching** reports each patch's raw effect (`patched − corrupted`) and its normalized value,
  that effect divided by the gap `clean − corrupted`. If the corruption doesn't change the metric
  (gap 0), normalized values are undefined: `PatchResult::normalized()` is false, and `indra patch`
  says so and shows raw effects instead of a table of NaN. If the gap is under 1 it warns, because
  dividing by a small gap amplifies noise. `.` in a table marks a site that doesn't exist (e.g.
  `mlp_out` on an attention-only layer).
- **Attribution patching** is a first-order estimate of patching (see above). It is reliable for
  ranking sites, not for their exact size. **Mixed patching** is exact at the sites it measures;
  the rest are estimates.
- **Mean ablation** replaces a head's output with its average over *reference* sequences: per
  position when they match the input's length, otherwise one pooled mean. It never averages over
  the input being tested. That would copy later tokens into earlier positions and break causality;
  earlier versions did this and could misjudge head importance by several times. Pass references
  with `--reference-file` (one per line), `--reference-text`, `--reference`, or
  `--reference-random N` (only meaningful for models trained on random tokens). In C++, see
  `head_means` and `ablate_head`. `ablate` measures next-token loss over the whole sequence, so for
  formats with unpredictable parts (like the arithmetic prompts) the per-head numbers include that
  noise.
- **Scope:** one sequence at a time, float32 arithmetic (weights in fp16 and fp16 GPU inputs with
  `--fp16`), CPU plus the GPU on macOS (`--device`). Only built-in
  indranet components can be exported.

## Examples

**Induction heads in a 2-layer model.** `python examples/induction_heads.py` trains the
`attn_only_2l` preset (with RoPE) on sequences containing a repeated segment of random length,
then finds the circuit from the cached patterns. It checks the two-hop "virtual attention"
`P1 @ P0` from each query to the token it should copy, and confirms with head ablations. On CPU it
takes about 40 seconds. In our run the layer-0 heads settled on a lag of 9 rather than the textbook
previous-token lag of 1. RoPE makes every lag equally easy, so check the two-hop score, not just
attention to position t−1. The walkthrough above takes the same model apart with `indra`.

**A transformer for arithmetic.** [examples/arithmetic/](examples/arithmetic/) trains a
3.2M-parameter model on fixed-width `+`, `−`, `×` problems, reaching 100% on 5-digit addition,
99.2–100% on subtraction and 98% on 2-digit multiplication (a larger 10.7M model reaches 80% on
3-digit multiplication). Its [README](examples/arithmetic/README.md) then reverse-engineers how it
adds with `indra`:
- layer-0 heads fetch the matching digits and the carry digits;
- the model reads its own previous output for chained carries;
- the MLPs compute the sum.

```bash
python examples/arithmetic/arithmetic.py train --out arith.pt     # ~3 min on an Apple-silicon GPU
python examples/arithmetic/arithmetic.py export arith.pt arith.safetensors
build/indra attn arith.safetensors --text "58291+76834=+521531" --layer 0
```

## Tests

```bash
make test       # C++ unit tests
pytest          # Python tests, golden tests against PyTorch, the indra CLI, the bindings and the
                # examples (the bindings' tests build the module; skipped without nanobind)
```

The C++ unit tests are in five groups, so part of the suite can run on its own:

| Group | Covers |
|---|---|
| `core` | JSON, tensors, matrix multiplies, vector math, weight shapes |
| `analysis` | per model type: cache and hooks, logit lens, DLA, patching, ablation, circuits |
| `grad` | gradients against finite differences, attribution and mixed patching |
| `fp16` | fp16 conversion and weights, and the analysis invariants rerun with fp16 weights |
| `gpu` | GPU work chunking and thresholds; on macOS also the kernels, resident passes, GPU attention and its backward |

```bash
make test GROUPS=gpu,fp16            # only these groups
make test FILTER=attention           # only tests whose name contains the text (combines with GROUPS)
build/tfi_tests --list               # the groups and names a selection would run
ctest --test-dir build-cmake -R gpu  # CMake builds register one test per group
pytest python/tests/test_hf.py       # the Python tests are split by file
```

## CI/CD

GitHub Actions runs on every push to `main` and every pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **cpp**: the engine built with make (`-Werror -pedantic`) and CMake, using GCC and clang on Linux
  and clang on macOS.
- **sanitizers**: the C++ unit tests, golden tests and CLI on AddressSanitizer +
  UndefinedBehaviorSanitizer builds, and the unit tests under ThreadSanitizer.
- **python**: the full test suite on Linux and macOS × Python 3.9 and 3.12 (the bindings included),
  plus an end-to-end run
  (train the induction model, export it, analyse it with `indra`).

To release, bump `version` in `python/pyproject.toml`, commit, then push a matching tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

[.github/workflows/release.yml](.github/workflows/release.yml) checks that the tag matches the
version and reruns CI. It then publishes a GitHub Release with `indra` binaries (CLI, static
library, headers) for Linux x86-64 and macOS arm64, plus the Python wheel and sdist. Tags with a
hyphen, such as `v0.2.0-rc1`, become pre-releases. Dependabot keeps the action versions up to date.

## Not included (yet)

A wheel with the compiled bindings (they are built in the repository), loading Hugging Face folders
directly in the engine (they go through the converter), sequences past
`original_max_position_embeddings` with `dynamic` or `longrope` RoPE scaling, Unigram and WordPiece
tokenizers, padding masks for batched prompts of different lengths, and mixture-of-experts layers.
