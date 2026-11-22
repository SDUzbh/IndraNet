# Testing and CI

[Back to the README](../README.md)

## What is checked

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

## CI and releases

GitHub Actions runs on every push to `main` and every pull request
([.github/workflows/ci.yml](../.github/workflows/ci.yml)):

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

[.github/workflows/release.yml](../.github/workflows/release.yml) checks that the tag matches the
version and reruns CI. It then publishes a GitHub Release with `indra` binaries (CLI, static
library, headers) for Linux x86-64 and macOS arm64, plus the Python wheel and sdist. Tags with a
hyphen, such as `v0.2.0-rc1`, become pre-releases. Dependabot keeps the action versions up to date.
