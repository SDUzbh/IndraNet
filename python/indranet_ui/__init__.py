"""indranet_ui: a local web page for exploring a model with the tfi engine.

    python -m indranet_ui model.safetensors [more models or folders] [--port 8765] [--fp16] [--open]

Python's standard library serves the page and a small JSON API (indranet_ui.server) over the tfi
bindings (build them with `make python`); the page is plain JavaScript and canvas, with nothing to
install or build. See README.md.
"""
from .server import Library, Session, make_handler, serve  # noqa: F401

__all__ = ["Library", "Session", "make_handler", "serve"]
