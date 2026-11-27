#!/usr/bin/env python3
"""Check that every #include in the library respects its layers (docs/architecture.md).

The layers and what each may use come from CMakeLists.txt (TFI_LAYERS, tfi_deps_<layer>). A file in src/<layer>/
may include headers of its own layer and of the layers it depends on: internal ones ("model/ops.h", or a header
next to it) and public ones (include/tfi/*.h, each owned by the layer that implements it, PUBLIC_OWNER below).
Public headers are held to the same rule. Prints each include that breaks it; exits 1 if any does.
Run from the repository root (ctest does): python3 tools/check_layers.py
"""
import pathlib
import re
import sys

# The layer whose sources implement each public header.
PUBLIC_OWNER = {
    "json.h": "support",
    "device.h": "support",  # set_num_threads in support; set_device in gpu, which may read it
    "tokenizer.h": "text",
    "tensor.h": "core",
    "weight.h": "core",
    "config.h": "core",
    "safetensors.h": "core",
    "model.h": "model",
    "analysis.h": "analysis",
    "generate.h": "generation",
    "runtime.h": "generation",
}


def read_layers(cmake: str):
    layers = re.search(r"set\(TFI_LAYERS ([^)]*)\)", cmake).group(1).split()
    deps = {}
    for layer in layers:
        m = re.search(rf"set\(tfi_deps_{layer}\b([^)]*)\)", cmake)
        if not m:
            sys.exit(f"CMakeLists.txt: no tfi_deps_{layer}")
        deps[layer] = set(m.group(1).split())
    return layers, deps


def layer_of(include: str, here: pathlib.Path, layers):
    if include.startswith("tfi/"):
        return PUBLIC_OWNER.get(include[4:], "?")
    first = include.split("/")[0]
    if first in layers:
        return first
    if "/" not in include and here.parent.name in layers:  # a header next to the including file
        return here.parent.name
    return "?"


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    layers, deps = read_layers((root / "CMakeLists.txt").read_text())
    files = []
    for layer in layers:
        files += [(layer, p) for p in sorted((root / "src" / layer).glob("*")) if p.suffix in (".h", ".cpp", ".mm")]
    for p in sorted((root / "include" / "tfi").glob("*.h")):
        files.append((PUBLIC_OWNER.get(p.name, "?"), p))

    bad = 0
    for layer, path in files:
        rel = path.relative_to(root)
        if layer == "?":
            print(f"{rel}: not assigned to a layer (tools/check_layers.py PUBLIC_OWNER)")
            bad += 1
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
            if not m:
                continue
            target = layer_of(m.group(1), path, layers)
            if target == "?":
                print(f"{rel}:{n}: {m.group(1)} belongs to no layer")
                bad += 1
            elif target != layer and target not in deps[layer]:
                print(f"{rel}:{n}: {layer} includes {m.group(1)} from {target}, a layer it may not use")
                bad += 1
    if bad:
        print(f"{bad} include(s) break the layers")
        return 1
    print(f"layers: {len(files)} files, every include within its layer or below")
    return 0


if __name__ == "__main__":
    sys.exit(main())
