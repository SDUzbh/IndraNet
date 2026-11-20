"""tfi: Python bindings for IndraNet's C++ interpretability engine.

Build them once with ``make python`` (needs CMake and ``pip install nanobind``), then::

    import tfi

    model = tfi.Model.load("model.safetensors")            # indranet.export or indranet_hf files
    tokens = model.tokenize("The capital of France is")
    logits, cache = model.run_with_cache(tokens, names=["blocks.0.attn.hook_pattern"])
    cache["blocks.0.attn.hook_pattern"].shape                # (heads, T, T), float32 numpy

    def zero_head_3(z, name):                                # hooks may edit activations
        z[:, 3, :] = 0                                       # hook_z is (T, heads, d_head)
    model.forward(tokens, hooks={"blocks.0.attn.hook_z": zero_head_3})

    tfi.lens_grid(model, tokens)                             # logit lens, every position
    tfi.direct_logit_attribution(model, tokens, pos=-1, target=model.tokenize(" Paris", False)[0])
    tfi.patch(model, clean, corrupt, "resid_pre", pos=-1, target=t, other=o, method="attribution")

    new = tfi.generate(model, "The capital of France is", max_new_tokens=20)   # KV cache; hooks= apply
    model.decode(new)                                                          # at every step

Arrays come back as float32 numpy arrays (no copy of the engine's buffers); results of the analyses are
dicts of arrays and lists. Engine calls release the GIL. The functions mirror include/tfi/analysis.h.
"""
try:
    from ._tfi import (  # noqa: F401
        KVCache, Model, Tokenizer, composition_random_baseline, composition_scores, device, direct_logit_attribution,
        generate, generate_batch, gpu_fp16, head_ablation, head_scores, head_weights, lens_grid, logit_lens, logit_metric,
        metric_gradients, next_token_loss, num_threads, ov_copying_scores, patch, read_metadata, set_device,
        set_gpu_fp16, set_num_threads,
    )
except ImportError as e:  # pragma: no cover - depends on the build
    raise ImportError(f"the tfi bindings are not built ({e}); run `make python` in the repository root "
                      "(needs CMake and `pip install nanobind`)") from e

__all__ = [
    "KVCache", "Model", "Tokenizer", "composition_random_baseline", "composition_scores", "device",
    "direct_logit_attribution", "generate", "generate_batch", "gpu_fp16", "head_ablation", "head_scores", "head_weights", "lens_grid",
    "logit_lens", "logit_metric", "metric_gradients", "next_token_loss", "num_threads", "ov_copying_scores",
    "patch", "read_metadata", "set_device", "set_gpu_fp16", "set_num_threads",
]
