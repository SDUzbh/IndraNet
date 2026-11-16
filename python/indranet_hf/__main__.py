"""python -m indranet_hf SOURCE OUT [--dtype auto|float32|float16|bfloat16] [--no-vocab] [--config-only]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SUPPORTED_MODEL_TYPES, Unsupported, convert, translate_config
from .convert import resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m indranet_hf",
                                 description="Convert a Hugging Face checkpoint into an IndraNet model file. "
                                             f"Model types: {', '.join(SUPPORTED_MODEL_TYPES)}.")
    ap.add_argument("source", help="checkpoint folder (config.json + *.safetensors) or Hugging Face repo id")
    ap.add_argument("out", nargs="?", help="output .safetensors file")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                    help="type of the weight matrices (default: as in the checkpoint)")
    ap.add_argument("--revision", help="Hub revision (branch, tag or commit) when SOURCE is a repo id")
    ap.add_argument("--no-vocab", action="store_true", help="don't store the tokenizer's token strings")
    ap.add_argument("--config-only", action="store_true", help="print the IndraNet config and write nothing")
    a = ap.parse_args(argv)
    try:
        if a.config_only:
            hf = json.loads((resolve(a.source, a.revision) / "config.json").read_text())
            print(json.dumps(translate_config(hf), indent=2))
            return 0
        if not a.out:
            ap.error("OUT is required unless --config-only")
        s = convert(a.source, a.out, dtype=a.dtype, vocab=not a.no_vocab, revision=a.revision)
    except (Unsupported, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    c = s["config"]
    print(f"{s['model_type']} -> {a.out}: {c['n_layers']} layers, d_model {c['d_model']}, {s['tensors']} tensors, "
          f"{s['bytes'] / 1e6:.1f} MB, {'tokenizer' if s['tokenizer'] else 'no tokenizer'}{'' if s['vocab'] else ', no vocabulary'}")
    print(f"next: build/indra info {Path(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
