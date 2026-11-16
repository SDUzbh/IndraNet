"""Export trained models (and reference activations) for the C++ interpretability tools.

Files are safetensors: an 8-byte little-endian header length, a JSON header, then raw
tensor bytes. The model's config is stored as JSON in the header metadata, so a single
file fully describes the model.

    from indranet.export import export_model
    export_model(model, "model.safetensors")                     # float32
    export_model(model, "model.safetensors", dtype="float16")    # 2-D weights in fp16: half the file
    export_model(model, "model.safetensors", dtype="bfloat16")   # 2-D weights in bf16: half the file

Then, from the repository root: ``build/indra info model.safetensors``.
"""
from __future__ import annotations

import json
import struct
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor

from .model import DecoderLM

FORMAT_VERSION = "1"
_DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64"}
FP16_MAX = 65504.0  # largest finite fp16 value

# Components the C++ side implements. Custom registered components cannot be exported.
SUPPORTED = {
    "norm": {"rmsnorm", "layernorm", "none"},
    "positional": {"rope", "alibi", "learned", "sinusoidal", "none"},
    "attention": {"causal"},
    "mlp": {"mlp", "glu", "swiglu", "geglu", "none"},
    "activation": {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"},
}


def save_safetensors(tensors: Dict[str, Tensor], path: Union[str, Path],
                     metadata: Optional[Dict[str, str]] = None) -> None:
    header: Dict[str, object] = {"__metadata__": dict(metadata or {})}
    blobs, offset = [], 0
    for name, t in tensors.items():
        t = t.detach().cpu().contiguous()
        if t.dtype not in _DTYPES:
            t = t.to(torch.float32)
        # numpy has no bfloat16: its values go out as their 16-bit patterns.
        data = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
        header[name] = {"dtype": _DTYPES[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def _check_supported(model: DecoderLM) -> None:
    cfg = model.cfg
    for i in range(cfg.n_layers):
        lc = cfg.layer_config(i)
        for kind, value in [("norm", lc.norm), ("positional", lc.positional), ("attention", lc.attention),
                            ("mlp", lc.mlp), ("activation", lc.activation)]:
            if value not in SUPPORTED[kind]:
                raise ValueError(f"layer {i}: {kind} '{value}' has no C++ implementation; "
                                 f"supported: {sorted(SUPPORTED[kind])}")


def export_model(model: DecoderLM, path: Union[str, Path], vocab: Optional[Sequence[str]] = None,
                 metadata: Optional[Dict[str, str]] = None, dtype: str = "float32") -> None:
    """Write weights and config. Tied unembedding weights are stored once.

    ``vocab`` (one string per token id) lets the C++ tools take ``--text`` input and print
    tokens as text. ``metadata`` adds free-form string entries to the file header.

    ``dtype="float16"`` stores every 2-D weight (embeddings, attention and MLP projections,
    unembedding) in half precision, which halves the file. Each value is rounded to the nearest
    fp16 value, as ``indra --fp16`` rounds a float32 file; norm scales and biases stay float32. A
    weight with a value outside fp16's range (|v| > 65504, or not finite) stays float32, with a
    warning. ``indra --fp16`` then loads these weights as they are, without converting them.

    ``dtype="bfloat16"`` does the same in bfloat16 (rounded to nearest even, as torch's
    ``.bfloat16()``): fp32's range, so only a value that rounds to infinity keeps a weight float32,
    but 8 significant bits instead of fp16's 11. ``indra`` loads these weights as float32, exactly;
    ``indra --fp16`` converts them to fp16, exactly for every value from about 7.6e-6 to 65504.
    """
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"dtype must be 'float32', 'float16' or 'bfloat16', got {dtype!r}")
    _check_supported(model)
    if vocab is not None and len(vocab) != model.cfg.vocab_size:
        raise ValueError(f"vocab has {len(vocab)} entries but the model's vocab_size is {model.cfg.vocab_size}")
    state = model.state_dict()
    if model.cfg.tie_embeddings:
        state.pop("unembed.weight", None)
    meta = dict(metadata or {})
    meta.update({"format": "indranet", "format_version": FORMAT_VERSION, "config": json.dumps(model.cfg.to_dict())})
    if vocab is not None:
        meta["vocab"] = json.dumps(list(vocab))
    tensors, kept = {}, []
    for name, value in state.items():
        value = value.detach().float()
        if dtype == "float16" and value.dim() == 2:
            if value.numel() == 0 or (torch.isfinite(value).all() and value.abs().max() <= FP16_MAX):
                value = value.half()
            else:
                kept.append(name)
        elif dtype == "bfloat16" and value.dim() == 2:
            rounded = value.bfloat16()
            if torch.isfinite(rounded).all():
                value = rounded
            else:
                kept.append(name)
        tensors[name] = value
    if kept:
        kind = "fp16" if dtype == "float16" else "bf16"
        warnings.warn(f"kept in float32 (values outside {kind}'s range): {', '.join(kept)}")
    save_safetensors(tensors, path, meta)


def hook_gradients(model: DecoderLM, tokens: Tensor, pos: int, target: int, other: int = -1) -> Dict[str, Tensor]:
    """d metric / d activation for every hook point, by autograd, for one sequence.

    The metric is ``logits[pos, target] - logits[pos, other]``, or the log-probability of ``target``
    when ``other`` is negative (tfi's ``logit_metric``). Each hook point is its own variable: e.g.
    ``hook_resid_pre``'s gradient counts every path from it, ``hook_attn_in``'s only the path
    through attention. Shapes are those of the cached activations, without the batch dimension.
    """
    if tokens.dim() == 1:
        tokens = tokens[None]
    acts: Dict[str, Tensor] = {}

    def capture(act: Tensor, name: str) -> Tensor:
        act = act.clone()  # a separate autograd node, even where a hook point returns its input
        if not act.requires_grad:  # e.g. sinusoidal positions, a constant buffer: a leaf to differentiate by
            act.requires_grad_()
        acts[name] = act
        return act

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad(), model.hooks([(None, capture)]):
            logits = model(tokens).logits[0, pos].float()
            metric = logits[target] - logits[other] if other >= 0 else logits.log_softmax(-1)[target]
            # Gradients with respect to the activations only: the parameters' .grad stay as they were.
            names = list(acts)
            found = torch.autograd.grad(metric, [acts[n] for n in names], allow_unused=True)
    finally:
        model.train(was_training)
    grads = {}
    for name, g in zip(names, found):
        g = g if g is not None else torch.zeros_like(acts[name])
        grads[name] = (g if name == "hook_pos_embed" else g[0]).detach().float()
    return grads


def export_reference(model: DecoderLM, tokens: Tensor, path: Union[str, Path],
                     metric: Optional[Sequence[int]] = None) -> None:
    """Write tokens, logits and every hook activation for one sequence, for golden tests.

    With ``metric=(pos, target, other)`` also write each hook point's gradient of that metric
    (see ``hook_gradients``) as ``grad.<name>``.
    """
    if tokens.dim() == 1:
        tokens = tokens[None]
    if tokens.shape[0] != 1:
        raise ValueError("reference export takes a single sequence")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out, cache = model.run_with_cache(tokens)
    finally:
        model.train(was_training)
    tensors = {"tokens": tokens[0].long(), "logits": out.logits[0].float()}
    for name, act in cache.items():
        # hook_pos_embed is (T, d) with no batch dimension; everything else is batch-first.
        tensors[f"cache.{name}"] = (act if name == "hook_pos_embed" else act[0]).float()
    meta = {"format": "indranet-reference"}
    if metric is not None:
        pos, target, other = (int(v) for v in metric)
        for name, g in hook_gradients(model, tokens, pos, target, other).items():
            tensors[f"grad.{name}"] = g
        meta["metric"] = json.dumps([pos, target, other])
    save_safetensors(tensors, path, meta)
