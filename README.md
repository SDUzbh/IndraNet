# IndraNet

[![CI](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml/badge.svg)](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml)

IndraNet takes trained decoder-only transformers apart. Its core is **tfi**, a dependency-free C++20
engine with named hook points, which runs the standard reverse-engineering methods: logit lens,
direct logit attribution, head scores, activation patching, ablation and weight-only circuit
analysis. It also generates text, with hooks at every step, to see what an intervention does to
what the model writes.

```
indranet (PyTorch)  --train-->  model  --export_model-->  model.safetensors  -->  indra (C++, tfi engine)
Hugging Face checkpoint  --indranet_hf-->  model.safetensors  -->  tfi (Python bindings)
```

## Quick start

```bash
make                                   # build/indra (CLI) and build/libtfi.a
pip install -e "python[dev]"           # the indranet package
python examples/induction_heads.py --export induction.safetensors   # train a small circuit
build/indra heads induction.safetensors
```

Or convert a Hugging Face checkpoint (Llama, Mistral, Qwen, Phi, Gemma, GPT-2, Pythia):

```bash
python -m indranet_hf path/to/checkpoint model.safetensors    # or a Hub repo id
build/indra generate model.safetensors --text "The capital of France is" --max-new 30
```

macOS builds use Accelerate and the GPU (macOS 26 or newer, built with Xcode 26); elsewhere a portable CPU path is used.
`--fp16` keeps the weights in half precision, and `--device cpu|both|gpu` picks where work runs.

## Commands

| Command | What it answers |
|---|---|
| `info` | Architecture, parameter count, hook point names |
| `run` | Next-token predictions and loss |
| `lens` | Logit lens: what the residual stream predicts after each sublayer |
| `dla` | Direct logit attribution: which heads and MLPs write the answer |
| `heads` | Per-head pattern scores (previous-token, induction, ...) and two-hop induction pairs |
| `patch` | Activation patching, per layer × position or per head: exact, attribution or mixed |
| `ablate` | Loss increase when each head is zero- or mean-ablated |
| `circuits` | Weights only: Q/K/V composition scores, OV copying |
| `attn` | Where each head of a layer attends, query by query |
| `tokenize` | Token ids for text |
| `generate` | Continue the input: greedy or sampled, with a trace, or speculative decoding |

Give tokens as `--tokens 5,17,3` or `--text "..."` (the model's own tokenizer for converted
models). `build/indra --help` lists every option.

## Example: the induction circuit

The model from `examples/induction_heads.py` (2 layers × 4 heads, attention-only, RoPE), taken apart
in five steps:

```
$ indra heads induction.safetensors --seg 20 --samples 16       # which heads form the circuit?
two-hop induction (reader attends to r, writer at r attends to the token to copy; any lag):
  L0H2 -> L1H3   0.920
  L0H0 -> L1H3   0.916

$ indra lens induction.safetensors --repeat-random 20 --seed 1 --pos 30    # where does the answer appear?
stage     p(target)  rank
L0.attn   0.0017     112
L1.attn   0.4076     0

$ indra dla induction.safetensors --repeat-random 20 --seed 1 --pos 31 --target 101 --other 99   # who writes it?
actual 7.2142   sum of parts 7.2142   (final norm scale frozen)
L1H3                2.4083  +########################################
L1H0                2.2056  +####################################

$ indra patch ... --site heads                                   # how does the information move?
             H0     H1     H2     H3
      L0  0.475 -0.009  0.549  0.001

$ indra circuits induction.safetensors                           # do the weights alone confirm it?
two-head OV path copying (writer OV, then reader OV):
  L0H2 -> L1H0   0.969
```

They agree on one mechanism: layer-0 heads (L0H0, L0H2) shift each token's information forward by
9 positions, and layer-1 heads match on it and copy the answer. RoPE makes every lag equally easy,
which is why the textbook previous-token score misses these heads. The
[arithmetic example](examples/arithmetic/README.md) does the same for a model that adds, subtracts and
multiplies.

## From code

**Python**: one command builds and installs the `tfi` module (it needs CMake and a C++20 compiler):

```bash
pip install git+https://github.com/SDUzbh/IndraNet     # or, in a checkout: pip install .
```

```python
import tfi
m = tfi.Model.load("model.safetensors", fp16=True)
t = m.tokenize("When Mary and John went to the store, John gave a drink to")
logits, cache = m.run_with_cache(t, names=["blocks.9.attn.hook_pattern"])   # numpy arrays
def knock_out(z, name):
    z[:, 7, :] = 0
logits = m.forward(t, hooks={"blocks.15.attn.hook_z": knock_out})
new = tfi.generate(m, t, max_new_tokens=20, hooks={"blocks.15.attn.hook_z": knock_out})
```

**C++ with CMake**: add the repository to your project and link `tfi::tfi`; include paths, C++20 and
the macOS frameworks come with it.

```cmake
include(FetchContent)
FetchContent_Declare(indranet GIT_REPOSITORY https://github.com/SDUzbh/IndraNet GIT_TAG main)
FetchContent_MakeAvailable(indranet)          # or add_subdirectory(path/to/IndraNet)
target_link_libraries(your_app PRIVATE tfi::tfi)
```

After `cmake --install`, `find_package(tfi)` gives the same target. **Without CMake**: `make install
PREFIX=$HOME/.local`, then `c++ app.cpp $(pkg-config --cflags --libs tfi)` (with
`PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig`).

```cpp
#include "tfi/analysis.h"
#include "tfi/generate.h"
using namespace tfi;

Model m = Model::load("model.safetensors");
Tokens t = {5, 17, 3, 5, 17};
ActivationCache cache;
Tensor logits = m.run_with_cache(t, cache);                 // every hook point
Hooks hooks;
hooks.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) { /* edit in place */ });
Tensor edited = m.forward(t, &hooks);
DLAResult dla = direct_logit_attribution(m, t, /*pos=*/4, /*target=*/3, /*other=*/9);
Tokens more = generate(m, t, GenerateOptions{}, &hooks);    // hooks apply at every step
```

The Python functions mirror the C++ API (`logit_lens`, `patch`, `head_ablation`, `generate`, ...).
The [indranet package](python/README.md) builds, trains and exports models in PyTorch with the same
hook names.

## More

| Page | Covers |
|---|---|
| [docs/api.md](docs/api.md) | Building (make, CMake, devices), the C++ API and the Python bindings in full |
| [docs/models.md](docs/models.md) | fp16 and bf16 files, memory mapping, the Hugging Face converter, the tokenizer |
| [docs/generation.md](docs/generation.md) | Sampling, the KV cache, `--trace`, batches, speculative decoding |
| [docs/methods.md](docs/methods.md) | The full induction walkthrough, attribution and mixed patching, every method's limits |
| [docs/testing.md](docs/testing.md) | What the tests check (golden tests against PyTorch and transformers), test groups, CI, releases |

| Path | What it is |
|---|---|
| [include/tfi/](include/tfi/), [src/](src/) | The tfi engine: its API, and the code in `model/`, `gpu/`, `text/`, `generation/` and `io/` |
| [tools/indra.cpp](tools/indra.cpp) | The `indra` command line |
| [python/](python/) | `indranet` (PyTorch models), [`indranet_hf`](python/indranet_hf/) (converter), [`tfi`](python/tfi/) (bindings) |
| [examples/](examples/) | Two circuits trained in Python and reverse-engineered with `indra` |
| [tests/](tests/) | C++ unit tests and the golden-test runner |

Tests: `make test` (C++) and `pytest` (Python, golden tests, CLI, bindings, examples).

**Not included yet:** loading Hugging Face folders directly in the engine (they go through the
converter), sequences past `original_max_position_embeddings` with `dynamic` or `longrope` RoPE
scaling, Unigram and WordPiece tokenizers, padding masks for batched prompts of different
lengths, mixture-of-experts layers, and a wheel with the compiled bindings.
