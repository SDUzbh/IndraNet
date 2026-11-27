# Architecture

[Back to the README](../README.md)

The engine, tfi, is one library built from layers. Each layer may use only the layers below it.
This is the same layering LLVM uses (Support, then IR, then CodeGen, then the tools), at a smaller
scale. The command line (`tools/indra.cpp`) and the Python bindings (`python/tfi/_tfi.cpp`) are thin
drivers over the public API in `include/tfi/`. Models reach the engine only as files: safetensors
and a config, written by `indranet.export` or the Hugging Face converter (`python/indranet_hf`).

## Layers

| Layer | Directory | What it holds | May use |
|---|---|---|---|
| support | `src/support/` | JSON, fp16 conversion, vector math, threads (`parallel.h`: `parallel_for`, `side_by_side`; `set_num_threads`) | nothing |
| text | `src/text/` | UTF-8, Unicode tables, the split-pattern regex, tokenizers | support |
| core | `src/core/` | tensors, the float32 matrix multiply, weights (fp32 and fp16, memory mapping), configs, safetensors files | support |
| gpu | `src/gpu/` | the Metal device (`gpu.mm`: kernels, `Gpu::Seq`), chunked GPU work, the device setting (`set_device`) | core, support |
| model | `src/model/` | `Model` and its hooks; the CPU forward pass and KV caches (`model.cpp`); the backward pass (`grad.cpp`); the GPU lowering of batched and gradient passes (`gpu_lowering.cpp`); where each pass runs (`placement.cpp`) | gpu, core, text, support |
| analysis | `src/analysis/` | logit lens, DLA, patching, attribution and mixed patching, ablation, circuits | model and below |
| generation | `src/generation/` | sampling, `generate`, speculative decoding, the prompt-cache runtime | analysis and below |

Each public header belongs to the layer that implements it: `json.h` and `device.h` to support,
`tokenizer.h` to text, `tensor.h`, `weight.h`, `config.h` and `safetensors.h` to core,
`model.h` to model, `analysis.h` to analysis, and `generate.h` and `runtime.h` to generation.
Internal headers live next to their sources and are included by path (`"model/ops.h"`). Those
paths are never part of the public API.

## How the layers are checked

Two checks back each other up. Both read the layers from `CMakeLists.txt` (`TFI_LAYERS`,
`tfi_deps_<layer>`):

- **Includes:** `tools/check_layers.py` (the ctest test `layers`) checks that every `#include` in
  `src/` and `include/tfi/` stays within its layer or below.
- **Links:** with `-DTFI_CHECK_LAYERS=ON`, CMake also builds each layer as a shared library linked
  only to the layers below it. A call into a layer above then fails to link, even when the caller
  declared the function itself. CI builds this on every push.

```bash
python3 tools/check_layers.py
cmake -S . -B build-layers -DTFI_CHECK_LAYERS=ON -DTFI_BUILD_TOOLS=OFF && cmake --build build-layers
```

A new source file goes in its layer's directory, and CMake and make pick it up there. A file
anywhere else in `src/` stops CMake with an error.

## CPU or GPU

Every rule that chooses between the CPU and the GPU for a pass lives in `src/model/placement.h`:
- single passes;
- cached and generated passes;
- the linear layers, unembedding, blocks and attention of batched passes;
- gradient passes;
- whether attribution patching's gradient runs use the GPU.

Analyses and the model ask placement instead of comparing sizes themselves. That way, passes that
must agree are always placed alike. Attribution patching's clean and corrupted runs are one
example: their metrics must equal exact patching's, bit for bit.

The GPU's own thresholds and capabilities (`gpu_min_rows`, `gpu_min_work`, which kernels a head
size can use) stay in `src/gpu/`. The tests' switches for these rules (`set_gpu_attention` and the
others) are in placement.h too.

The GPU computes what the CPU computes. `gpu_lowering.cpp` records each block, and its backward
pass, as `Gpu::Seq` operations, from the same weights and config as `Model::block` and grad.cpp's
backward pass. The unit tests hold the two to each other (see [testing.md](testing.md)).
