"""Token strings for display, read from a Hugging Face tokenizer.json (or GPT-2's vocab.json).

These let IndraNet's tools print tokens as text. They are not a tokenizer: the engine's ``--text``
matches the longest token at each point, which can split text differently from the model's BPE.
For exact token ids, tokenize with the model's own tokenizer and pass the ids.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional


def _byte_decoder() -> Dict[str, int]:
    """GPT-2's byte-level alphabet: the printable character each byte is written as, reversed."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTES = _byte_decoder()
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def byte_level(token: str) -> str:
    if not all(ch in _BYTES for ch in token):
        return token
    return bytes(_BYTES[ch] for ch in token).decode("utf-8", errors="backslashreplace")


def sentencepiece(token: str) -> str:
    m = _BYTE_TOKEN.fullmatch(token)
    if m:
        b = int(m.group(1), 16)
        return chr(b) if 0x20 <= b < 0x7F else token
    return token.replace("▁", " ")


def _is_byte_level(tok: dict) -> bool:
    def has(node) -> bool:
        if not isinstance(node, dict):
            return False
        if node.get("type") == "ByteLevel":
            return True
        return any(has(n) for key in ("decoders", "pretokenizers") for n in node.get(key) or [])
    return has(tok.get("decoder")) or has(tok.get("pre_tokenizer"))


def load(folder: Path, vocab_size: int) -> Optional[List[str]]:
    """One display string per token id, or None when the folder has no tokenizer we can read."""
    added: Dict[int, str] = {}
    if (folder / "tokenizer.json").exists():
        tok = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
        model = tok.get("model") or {}
        vocab = model.get("vocab")
        if isinstance(vocab, list):  # Unigram: [[piece, score], ...]
            ids = {piece: i for i, (piece, _) in enumerate(vocab)}
        elif isinstance(vocab, dict):
            ids = vocab
        else:
            warnings.warn("tokenizer.json has no vocabulary this converter reads; writing no vocabulary")
            return None
        decode = byte_level if _is_byte_level(tok) else sentencepiece
        added = {a["id"]: a["content"] for a in tok.get("added_tokens") or []}
    elif (folder / "vocab.json").exists():  # GPT-2's older files
        ids, decode = json.loads((folder / "vocab.json").read_text(encoding="utf-8")), byte_level
    else:
        return None
    out = [f"<|{i}|>" for i in range(vocab_size)]  # ids the tokenizer never produces (padding rows)
    dropped = 0
    for piece, i in ids.items():
        if 0 <= i < vocab_size:
            out[i] = decode(piece)
        else:
            dropped += 1
    for i, content in added.items():  # special and added tokens: their text as it is
        if 0 <= i < vocab_size:
            out[i] = content
        else:
            dropped += 1
    if dropped:
        warnings.warn(f"{dropped} tokenizer entries have ids beyond the model's {vocab_size} rows; left out")
    return out


# What IndraNet's engine implements of tokenizer.json (include/tfi/tokenizer.h).
_SUPPORTED = {
    "normalizer": {"NFC", "NFD", "Replace", "Prepend", "Sequence"},
    "pre_tokenizer": {"ByteLevel", "Split", "Metaspace", "Digits", "Sequence"},
    "post_processor": {"TemplateProcessing", "ByteLevel", "RobertaProcessing", "BertProcessing", "Sequence"},
}


def _unsupported(kind: str, node) -> Optional[str]:
    if not node:
        return None
    if node.get("type") not in _SUPPORTED[kind]:
        return f"{kind} {node.get('type')!r}"
    for child in node.get("normalizers") or node.get("pretokenizers") or node.get("processors") or []:
        why = _unsupported(kind, child)
        if why:
            return why
    return None


def tokenizer(folder: Path) -> Optional[dict]:
    """The parts of the folder's tokenizer the engine uses (normalizer, pre-tokenizer, BPE model,
    added tokens, post-processor), or None when there is none, or it uses something the engine doesn't
    implement (with a warning: --text then falls back to longest-match on the vocabulary). GPT-2's
    older vocab.json + merges.txt become the byte-level BPE tokenizer they describe."""
    if (folder / "tokenizer.json").exists():
        tok = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    elif (folder / "vocab.json").exists() and (folder / "merges.txt").exists():
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        merges = [l for l in (folder / "merges.txt").read_text(encoding="utf-8").split("\n")
                  if l and not l.startswith("#version")]
        tok = {"model": {"type": "BPE", "vocab": vocab, "merges": merges},
               "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
               "added_tokens": [{"id": vocab["<|endoftext|>"], "content": "<|endoftext|>", "special": True}]
               if "<|endoftext|>" in vocab else []}
    else:
        return None
    model = tok.get("model") or {}
    why = None
    if model.get("type", "BPE") != "BPE":
        why = f"model {model.get('type')!r} (only BPE is implemented)"
    elif model.get("continuing_subword_prefix") or model.get("end_of_word_suffix") or model.get("dropout"):
        why = "BPE continuing_subword_prefix / end_of_word_suffix / dropout"
    for kind in _SUPPORTED:
        why = why or _unsupported(kind, tok.get(kind))
    if why:
        warnings.warn(f"tokenizer not stored: its {why} is not implemented by the engine; --text will match the "
                      "longest vocabulary entry instead, so pass token ids for exact inputs")
        return None
    return {k: tok.get(k) for k in ("normalizer", "pre_tokenizer", "model", "post_processor", "added_tokens")}
