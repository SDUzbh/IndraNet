"""Convert a Hugging Face checkpoint folder (config.json + *.safetensors) into an IndraNet model file."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from . import families, tensors, vocab as vocab_mod
from .families import IGNORED, Out, Unsupported

FORMAT_VERSION = "1"  # the IndraNet model file version this writes (indranet.export.FORMAT_VERSION)
DTYPES = {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}


def translate_config(hf: dict) -> dict:
    """IndraNet's config for a Hugging Face config.json (as a dict). ``bias`` is settled by
    ``convert``, which sees which bias tensors the checkpoint has."""
    return families.family(hf).config(hf)


def resolve(source: Union[str, Path], revision: Optional[str] = None) -> Path:
    """A local folder as it is; otherwise a Hub repo id, downloaded (configs, tokenizer, safetensors)."""
    p = Path(source)
    if p.is_dir():
        return p
    if p.exists() or not re.fullmatch(r"[\w.-]+/[\w.-]+", str(source)):
        raise FileNotFoundError(f"{source}: not a checkpoint folder or a Hugging Face repo id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(f"{source} is not a local folder; to download it from the Hugging Face Hub, "
                          "pip install huggingface_hub (or download the folder yourself)") from e
    return Path(snapshot_download(str(source), revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"]))


def shard_files(folder: Path) -> List[Path]:
    index = folder / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return [folder / f for f in sorted(set(weight_map.values()))]
    if (folder / "model.safetensors").exists():
        return [folder / "model.safetensors"]
    files = sorted(folder.glob("*.safetensors"))
    if files:
        return files
    if list(folder.glob("*.bin")) or list(folder.glob("*.pt")):
        raise Unsupported(f"{folder} has PyTorch pickle weights only. They aren't read here because loading a pickle "
                          "can run code; save the model with safe_serialization=True (the default) first")
    raise FileNotFoundError(f"{folder} has no *.safetensors files")


def _plan(cfg: dict, outs: Dict[str, Out], shards: tensors.Shards, dtype: str):
    """(dtype, shape, produce) per output tensor, and the 2-D weights kept float32 because a value
    doesn't fit the asked type."""
    plan, kept = {}, []
    for name, o in outs.items():
        src_dtype = shards.dtype(o.src)

        def raw(o=o):
            a = shards.get(o.src)
            return o.pick(a) if o.pick else a

        a = raw()
        if a.ndim != 2:  # norms, biases: float32, as indranet.export writes them
            def produce(raw=raw, d=src_dtype, add=o.add):
                f = tensors.to_f32(raw(), d)
                return f + np.float32(add) if add else f
            plan[name] = ("F32", a.shape, produce)
            continue
        target = "F32" if src_dtype == "F64" else src_dtype if dtype == "auto" else DTYPES[dtype]
        if target != src_dtype and target != "F32" and tensors.cast(a, src_dtype, target) is None:
            kept.append(name)
            target = "F32"
        plan[name] = (target, a.shape, lambda raw=raw, d=src_dtype, t=target: tensors.cast(raw(), d, t))
    return plan, kept


def convert(source: Union[str, Path], out: Union[str, Path], dtype: str = "auto", vocab: bool = True,
            revision: Optional[str] = None) -> dict:
    """Write an IndraNet model file for a Hugging Face checkpoint.

    ``source`` is a folder with config.json and the weights as safetensors (one file or shards), or a
    Hub repo id (downloaded with huggingface_hub). ``dtype`` is the type of the 2-D weights:
    ``"auto"`` keeps the checkpoint's (bf16 stays bf16), or ``"float32"``, ``"float16"``,
    ``"bfloat16"``; a weight with a value that doesn't fit the asked type stays float32, with a
    warning. Norms and biases are always float32. With ``vocab`` the tokenizer's token strings are
    stored for display. Returns a summary: the IndraNet config, tensor count, file size and the
    weights kept float32.
    """
    if dtype != "auto" and dtype not in DTYPES:
        raise ValueError(f"dtype must be 'auto', 'float32', 'float16' or 'bfloat16', got {dtype!r}")
    folder = resolve(source, revision)
    hf = json.loads((folder / "config.json").read_text())
    fam = families.family(hf)
    cfg = fam.config(hf)
    shards = tensors.Shards(shard_files(folder))
    names = set(shards.names())
    outs = fam.tensors(cfg, names)

    missing = sorted({o.src for o in outs.values()} - names)
    if missing:
        raise Unsupported(f"the checkpoint is missing {len(missing)} tensors the config implies, e.g. {missing[:3]}")
    used = {o.src for o in outs.values()}
    tied_copies = {n for n in names if cfg["tie_embeddings"] and n.endswith(("lm_head.weight", "embed_out.weight"))}
    leftover = sorted(n for n in names - used - tied_copies if not IGNORED.search(n))
    if leftover:
        raise Unsupported(f"{len(leftover)} tensors have no IndraNet counterpart (a variant this converter doesn't "
                          f"know?), e.g. {leftover[:3]}")
    if shards.shape(outs["embed.weight"].src)[0] != cfg["vocab_size"]:
        raise Unsupported(f"the embedding has {shards.shape(outs['embed.weight'].src)[0]} rows, "
                          f"but config.json says vocab_size={cfg['vocab_size']}")
    cfg["bias"] = families.bias_setting(cfg, outs)

    plan, kept = _plan(cfg, outs, shards, dtype)
    if kept:
        warnings.warn(f"kept in float32 (values outside the asked type's range): {', '.join(kept)}")
    meta = {"format": "indranet", "format_version": FORMAT_VERSION, "config": json.dumps(cfg),
            "source": "huggingface", "hf_model_type": hf["model_type"], "hf_config": json.dumps(hf)}
    if vocab:
        v = vocab_mod.load(folder, cfg["vocab_size"])
        if v is not None:
            meta["vocab"] = json.dumps(v)
    size = tensors.write(Path(out), plan, meta)
    return {"config": cfg, "tensors": len(plan), "bytes": size, "kept_float32": kept,
            "vocab": "vocab" in meta, "model_type": hf["model_type"]}
