"""Safetensors reading and writing with numpy only, and the float conversions the converter needs.

numpy has no bfloat16, so bf16 tensors are carried as uint16 arrays of their bit patterns. Moving,
slicing and transposing them needs no conversion; values are converted only where asked.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# safetensors dtype -> numpy dtype it is held in (BF16: its bits, as uint16)
NUMPY = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "F64": np.float64, "I64": np.int64, "U8": np.uint8}
# Matrices start on this boundary in files write() makes (16 KiB: Apple silicon's page, a multiple of
# 4 KiB ones), so the engine can memory-map them and use them in place, the GPU included.
ALIGN = 16384
PADDING = "__padding"  # prefix of the filler tensors that keep the data free of holes, as safetensors requires
FP16_MAX = 65504.0


class Shards:
    """The tensors of one or more safetensors files, read lazily through memory maps."""

    def __init__(self, files: Iterable[Path]):
        self._where: Dict[str, Tuple[Path, dict, int]] = {}
        self._maps: Dict[Path, np.memmap] = {}
        for f in files:
            f = Path(f)
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            header.pop("__metadata__", None)
            for name, info in header.items():
                if name in self._where:
                    raise ValueError(f"tensor {name!r} appears in both {self._where[name][0].name} and {f.name}")
                self._where[name] = (f, info, 8 + n)

    def names(self) -> List[str]:
        return [n for n in self._where if not n.startswith(PADDING)]

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def dtype(self, name: str) -> str:
        return self._where[name][1]["dtype"]

    def shape(self, name: str) -> Tuple[int, ...]:
        return tuple(self._where[name][1]["shape"])

    def get(self, name: str) -> np.ndarray:
        """The tensor as stored (BF16 as uint16 bits), backed by the file: nothing is read until used."""
        f, info, base = self._where[name]
        if info["dtype"] not in NUMPY:
            raise ValueError(f"tensor {name!r} has dtype {info['dtype']}, which the converter does not read")
        if f not in self._maps:
            self._maps[f] = np.memmap(f, dtype=np.uint8, mode="r")
        lo, hi = info["data_offsets"]
        raw = self._maps[f][base + lo:base + hi]
        return raw.view(NUMPY[info["dtype"]]).reshape(info["shape"])


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact: a bf16 value is the top half of the float32 with the same value."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest, ties to even (as torch's .bfloat16()). Callers check the inputs are finite."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def to_f32(a: np.ndarray, dtype: str) -> np.ndarray:
    return bf16_to_f32(a) if dtype == "BF16" else np.asarray(a, dtype=np.float32)


def cast(a: np.ndarray, dtype: str, target: str) -> Optional[np.ndarray]:
    """``a`` (held as ``dtype``) in ``target``, or None when a value doesn't fit there (not finite, or
    outside fp16's range, or rounding to bf16's infinity)."""
    if dtype == target:
        return a
    if target == "F32":
        return to_f32(a, dtype)
    if target == "BF16" and dtype == "F64":
        a, dtype = np.asarray(a, dtype=np.float32), "F32"
    f = to_f32(a, dtype)
    if not np.isfinite(f).all():
        return None
    if target == "F16":
        return f.astype(np.float16) if (f.size == 0 or np.abs(f).max() <= FP16_MAX) else None
    if target == "BF16":
        b = f32_to_bf16(f)
        return None if ((b & 0x7F80) == 0x7F80).any() else b
    raise ValueError(f"unknown target dtype {target}")


def write(path: Path, tensors: Dict[str, Tuple[str, Tuple[int, ...], "callable"]],
          metadata: Dict[str, str], align: int = ALIGN) -> int:
    """Write a safetensors file. ``tensors`` maps each name to (dtype, shape, produce), where
    produce() returns the array; tensors are produced one at a time, in order, so only one is in
    memory. Returns the file size.

    The data starts at a multiple of ``align`` bytes from the file's start, and so does every 2-D
    tensor (a matrix): small ``__padding<k>`` U8 tensors fill the gaps, since safetensors allows no
    holes in the data. Readers that know the convention skip them; to others they are harmless extra
    tensors. ``align=1`` writes the tensors back to back.

    The file is written next to ``path`` under a temporary name and renamed to ``path`` only when it is
    complete, so a failure or an interruption part way leaves an existing file at ``path`` as it was."""
    entries = []  # (name, dtype, shape, produce or None for padding, offset, size)
    offset = 0
    for name, (dtype, shape, produce) in tensors.items():
        n = int(np.prod(shape, dtype=np.int64)) * np.dtype(NUMPY[dtype]).itemsize
        if len(shape) == 2 and offset % align:
            pad = -offset % align
            entries.append((f"{PADDING}{len(entries)}", "U8", (pad,), None, offset, pad))
            offset += pad
        entries.append((name, dtype, tuple(shape), produce, offset, n))
        offset += n
    header: Dict[str, object] = {"__metadata__": metadata}
    for name, dtype, shape, _, at, n in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + n]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % max(align, 8))  # spaces may pad the header; the data starts aligned
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            for name, dtype, shape, produce, _, n in entries:
                if produce is None:
                    f.write(bytes(n))
                    continue
                a = np.ascontiguousarray(produce(), dtype=NUMPY[dtype])
                if a.shape != tuple(shape):
                    raise AssertionError(f"{name}: produced shape {a.shape}, planned {tuple(shape)}")
                f.write(a.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:  # KeyboardInterrupt too: no half-written file left behind
        tmp.unlink(missing_ok=True)
        raise
    return 8 + len(raw) + offset
