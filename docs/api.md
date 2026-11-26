# Building and the APIs

[Back to the README](../README.md)

## Build

```bash
make            # build/indra (CLI), build/libtfi.a, build/tfi_tests, build/tfi_golden
make test       # C++ unit tests
make python     # the Python bindings in place, python/tfi/_tfi*.so, for working on them
make install    # libtfi.a, the headers and tfi.pc under PREFIX (/usr/local by default)
```

Or with CMake: `cmake -S . -B build-cmake && cmake --build build-cmake`. On macOS, matrix
multiplies use Accelerate and the GPU; elsewhere a portable loop is used. `indra --device` picks
where work runs: `both` (the default) sends work to the GPU when it is large enough to repay it,
`cpu` never uses the GPU (also `--no-gpu`), and `gpu` runs everything it can on the GPU, attention
included. macOS builds need macOS 26 or newer and its SDK (Xcode 26).

## C++ API

```cpp
#include "tfi/analysis.h"
#include "tfi/generate.h"
#include "tfi/runtime.h"
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
PromptCache pc;                                              // keys and values kept between calls: the next
opt.prompt_cache = &pc;                                      // call runs only what its prompt adds (pc.reused)
Tokens fast = generate(m, t, opt);
DecodeRuntime runtime(m, t, opt, &hooks);                    // tfi/runtime.h: the same decoding, a token at a time
while (std::optional<int64_t> token = runtime.next()) { /* ... */ }
std::string text = m.decode(more);

const Weight& Wq = m.weight("blocks.0.attn.q_proj.weight");  // float32 or fp16
Tensor q_rows = Wq.rows(0, 16);                              // read as float32 either way (also cols, to_tensor)
```

`Model::weight` returns a `Weight`. Matrices (float32 or fp16) are read with `rows()`, `cols()`,
`to_tensor()` and the `linear()` / `matmul()` overloads, which give float32 either way; they sit in
page-aligned memory that the GPU uses in place, so no weight is held twice. Norm scales and biases
are float32 tensors, read with `f32()`.

To use the engine from your own C++ project, link the CMake target `tfi::tfi`: from
`FetchContent_Declare(indranet GIT_REPOSITORY https://github.com/SDUzbh/IndraNet GIT_TAG main)` and
`FetchContent_MakeAvailable(indranet)`, from `add_subdirectory(path/to/IndraNet)`, or from
`find_package(tfi)` after `cmake --install`. It brings the include path, C++20 and (on macOS)
Accelerate and Metal with it; the CLI and the tests are built only when IndraNet is the top-level
project (`-DTFI_BUILD_TOOLS=ON` builds them anyway). Without CMake, `make install PREFIX=...` writes a
pkg-config file: `c++ app.cpp $(pkg-config --cflags --libs tfi)`.

## Python bindings

`pip install git+https://github.com/SDUzbh/IndraNet` (or `pip install .` in a checkout) builds the
module `tfi` with CMake and [nanobind](https://github.com/wjakob/nanobind) and installs it. To work on
the bindings, `make python` builds it in place in [python/tfi/](../python/tfi/), where `pytest` finds
it (it puts `python/` on the path).

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
pc = tfi.PromptCache()                     # kept between calls: the next one runs only what its prompt adds
tfi.generate(m, t + new + more, prompt_cache=pc)
cache = tfi.KVCache()                      # step by step: forward_cached runs only the new tokens
logits = m.forward_cached(t, cache)
logits = m.forward_cached([mary], cache)   # (1, vocab); hooks see the new rows
```

The functions mirror the C++ API (`logit_lens`, `head_scores`, `patch`, `metric_gradients`,
`head_ablation`, `composition_scores`, `ov_copying_scores`, `head_weights`, `generate`, `set_device`, ...).
Activations come back as numpy arrays that take over the engine's buffers, with no copy, and
analysis results as dicts. Every engine call releases the GIL, so threads can run it side by side.
