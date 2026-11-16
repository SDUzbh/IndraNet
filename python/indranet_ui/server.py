"""The UI's HTTP server: Python's standard library only, on top of the tfi bindings.

GET serves the page (static/); POST /api/<name> takes and returns JSON, except /api/attention, which
returns the (heads, T, T) pattern as raw little-endian float32 (shape in the X-Shape header). The server
offers a list of model files (a Library) and keeps one of them loaded, the first at start; the page can
switch to another (/api/models, /api/select). Engine calls are serialized with a lock (the engine
releases the GIL, so the server keeps answering while one runs).
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
MAX_TOKENS = 2048
PATTERN_CACHE_BYTES = 256 << 20  # every layer's attention for one prompt is kept when it fits this


class BadRequest(Exception):
    pass


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def topk(p: np.ndarray, k: int) -> np.ndarray:
    k = min(k, p.shape[-1])
    idx = np.argpartition(-p, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(p, idx, -1), axis=-1)
    return np.take_along_axis(idx, order, -1)


def finite(x) -> Optional[float]:
    x = float(x)
    return x if np.isfinite(x) else None


def matrix(a: np.ndarray) -> List[List[Optional[float]]]:
    """A 2-D array as JSON rows, NaN as null."""
    return [[finite(v) for v in row] for row in np.asarray(a, dtype=np.float64)]


class Session:
    """The loaded model and what the API computes from it."""

    def __init__(self, path: str, fp16: bool = False):
        import tfi
        self.tfi = tfi
        self.path = path
        self.meta = tfi.read_metadata(path)
        self.model = tfi.Model.load(path, fp16=fp16)
        self.lock = threading.Lock()
        self._patterns: Tuple[Optional[tuple], Dict[int, np.ndarray]] = (None, {})
        self._head_scores: Dict[tuple, dict] = {}

    # --- helpers -----------------------------------------------------------------------
    def token_text(self, i: int) -> str:
        return self.model.token_str(int(i))

    def tokens(self, body: dict, key: str = "tokens") -> List[int]:
        t = body.get(key)
        if not isinstance(t, list) or not t or not all(isinstance(x, int) for x in t):
            raise BadRequest(f"'{key}' must be a non-empty list of token ids")
        if len(t) > min(MAX_TOKENS, self.model.max_seq_len):
            raise BadRequest(f"{len(t)} tokens: at most {min(MAX_TOKENS, self.model.max_seq_len)}")
        V = self.model.vocab_size
        if any(x < 0 or x >= V for x in t):
            raise BadRequest(f"token ids must be in [0, {V})")
        return t

    def token_id(self, value: Any, what: str, required: bool = True) -> int:
        """A token given as an id (a number, or text "#123"), or as text that is exactly one token."""
        if value is None or value == "":
            if required:
                raise BadRequest(f"give a {what} token")
            return -1
        if isinstance(value, bool):
            raise BadRequest(f"{what} must be a token id or text")
        if isinstance(value, int) or (isinstance(value, str) and value[:1] == "#" and value[1:].isdigit()):
            i = int(value if isinstance(value, int) else value[1:])
            if not 0 <= i < self.model.vocab_size:
                raise BadRequest(f"{what} token id {i} is out of range")
            return i
        if not isinstance(value, str):
            raise BadRequest(f"{what} must be a token id or text")
        if not (self.model.has_tokenizer or self.model.vocab):
            raise BadRequest(f"this model has no vocabulary: give the {what} as a token id, written #123")
        ids = self.model.tokenize(value, False)
        if len(ids) != 1:
            shown = " + ".join(repr(self.token_text(i)) for i in ids) or "nothing"
            raise BadRequest(f"{what} {value!r} is {len(ids)} tokens ({shown}); give a single token")
        return ids[0]

    def pos(self, body: dict, T: int) -> int:
        p = body.get("pos", -1)
        if not isinstance(p, int) or not -T <= p < T:
            raise BadRequest(f"pos must be in [-{T}, {T})")
        return p % T

    # --- API -----------------------------------------------------------------------------
    def info(self, body: dict) -> dict:
        m = self.model
        cfg = json.loads(self.meta.get("config", "{}"))
        return {
            "file": Path(self.path).name,
            "source": self.meta.get("source"),
            "model_type": self.meta.get("hf_model_type"),
            "n_layers": m.n_layers, "d_model": m.d_model, "vocab_size": m.vocab_size, "max_seq_len": m.max_seq_len,
            "layer_heads": m.layer_heads, "num_parameters": m.num_parameters, "weight_bytes": m.weight_bytes,
            "fp16": m.fp16, "has_tokenizer": m.has_tokenizer, "has_vocab": bool(m.vocab),
            "device": self.tfi.device(), "config": cfg, "max_tokens": min(MAX_TOKENS, m.max_seq_len),
        }

    def tokenize(self, body: dict) -> dict:
        text = body.get("text", "")
        if not isinstance(text, str) or not text:
            raise BadRequest("give some text")
        if not (self.model.has_tokenizer or self.model.vocab):
            raise BadRequest("this model has no tokenizer or vocabulary: give token ids (comma-separated)")
        with self.lock:
            ids = self.model.tokenize(text, bool(body.get("special", True)))
        if not ids:
            raise BadRequest("the text has no tokens")
        return {"tokens": ids, "texts": [self.token_text(i) for i in ids]}

    def texts(self, body: dict) -> dict:
        return {"texts": [self.token_text(i) for i in self.tokens(body)]}

    def run(self, body: dict) -> dict:
        t = self.tokens(body)
        k = int(body.get("k", 5))
        with self.lock:
            logits = self.model.forward(t)
        p = softmax(logits.astype(np.float64))
        top = topk(p, k)
        T = len(t)
        nxt = [finite(p[i, t[i + 1]]) for i in range(T - 1)] + [None]
        rank = [int((p[i] > p[i, t[i + 1]]).sum()) for i in range(T - 1)] + [None]
        loss = float(-np.mean(np.log(np.maximum([p[i, t[i + 1]] for i in range(T - 1)], 1e-45)))) if T > 1 else None
        return {
            "texts": [self.token_text(i) for i in t],
            "top": [[[int(j), self.token_text(j), float(p[i, j])] for j in top[i]] for i in range(T)],
            "next_prob": nxt, "next_rank": rank, "loss": loss,
        }

    def attention(self, body: dict) -> Tuple[np.ndarray, Tuple[int, ...]]:
        t = self.tokens(body)
        layer = body.get("layer")
        if not isinstance(layer, int) or not 0 <= layer < self.model.n_layers:
            raise BadRequest(f"layer must be in [0, {self.model.n_layers})")
        key = tuple(t)
        with self.lock:
            cached_key, patterns = self._patterns
            if cached_key != key:
                patterns = {}
                self._patterns = (key, patterns)
            if layer not in patterns:
                T, L = len(t), self.model.n_layers
                every = sum(self.model.layer_heads) * T * T * 4 <= PATTERN_CACHE_BYTES
                layers = range(L) if every else [layer]
                names = [f"blocks.{i}.attn.hook_pattern" for i in layers]
                _, cache = self.model.run_with_cache(t, names=names)
                for i in layers:
                    patterns[i] = cache[f"blocks.{i}.attn.hook_pattern"]
            a = patterns[layer]
        return a, a.shape

    def lens(self, body: dict) -> dict:
        t = self.tokens(body)
        k = max(1, min(int(body.get("k", 5)), 20))
        with self.lock:
            g = self.tfi.lens_grid(self.model, t, k)
        tops = g["top_tokens"]
        words = {int(i): self.token_text(i) for i in np.unique(tops)}
        return {
            "stages": g["stages"],
            "top_tokens": tops.tolist(),
            "top_texts": [[[words[int(i)] for i in cell] for cell in row] for row in tops],
            "top_probs": np.round(g["top_probs"].astype(np.float64), 5).tolist(),
            "next_prob": matrix(g["next_prob"]),
            "next_rank": g["next_rank"].tolist(),
        }

    def dla(self, body: dict) -> dict:
        t = self.tokens(body)
        pos = self.pos(body, len(t))
        with self.lock:
            target = body.get("target")
            if target in (None, ""):  # the model's top prediction
                target = int(np.argmax(self.model.forward(t)[pos]))
            target = self.token_id(target, "target")
            other = self.token_id(body.get("other"), "other", required=False)
            r = self.tfi.direct_logit_attribution(self.model, t, pos, target, other)
        return {"labels": r["labels"], "values": [finite(v) for v in r["values"]], "sum": r["sum"], "actual": r["actual"],
                "pos": pos, "target": target, "target_text": self.token_text(target),
                "other": other, "other_text": self.token_text(other) if other >= 0 else None}

    def patch(self, body: dict) -> dict:
        clean, corrupt = self.tokens(body, "clean"), self.tokens(body, "corrupt")
        if len(clean) != len(corrupt):
            raise BadRequest(f"clean and corrupted inputs must have the same number of tokens ({len(clean)} vs {len(corrupt)})")
        pos = self.pos(body, len(clean))
        site = body.get("site", "resid_pre")
        if site not in ("resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out", "heads"):
            raise BadRequest(f"unknown site {site!r}")
        method = body.get("method", "exact")
        steps, verify = int(body.get("steps", 1)), int(body.get("verify", 10))
        with self.lock:
            target = self.token_id(body.get("target"), "target")
            other = self.token_id(body.get("other"), "other", required=False)
            r = self.tfi.patch(self.model, clean, corrupt, site, pos, target, other, method, steps, verify)
        out = {
            "row_labels": r["row_labels"], "col_labels": r["col_labels"],
            "effect": matrix(r["effect"]), "values": matrix(r["values"]) if r["normalized"] else None,
            "clean_metric": r["clean_metric"], "corrupt_metric": r["corrupt_metric"], "normalized": r["normalized"],
            "target": target, "target_text": self.token_text(target),
            "other": other, "other_text": self.token_text(other) if other >= 0 else None,
        }
        if "exact" in r:
            out["exact"] = r["exact"].tolist()
        return out

    def heads(self, body: dict) -> dict:
        seg, n = int(body.get("segment_len", 24)), int(body.get("n_samples", 8))
        if not (2 <= seg <= 512 and 1 <= n <= 64):
            raise BadRequest("segment_len must be in [2, 512] and n_samples in [1, 64]")
        if 2 * seg + 1 > self.model.max_seq_len:
            raise BadRequest(f"segment_len {seg} is too long for max_seq_len {self.model.max_seq_len}")
        key = (seg, n)
        with self.lock:
            if key not in self._head_scores:
                s = self.tfi.head_scores(self.model, seg, n, 0)
                self._head_scores[key] = {
                    **{name: matrix(s[name]) for name in ("previous_token", "induction", "duplicate_token", "first_token", "self")},
                    "top_two_hop": [[w, r, float(v)] for w, r, v in s["top_two_hop"]],
                }
            return self._head_scores[key]

    def routes(self) -> Dict[str, Callable[[dict], Any]]:
        return {"info": self.info, "tokenize": self.tokenize, "texts": self.texts, "run": self.run,
                "attention": self.attention, "lens": self.lens, "dla": self.dla, "patch": self.patch, "heads": self.heads}


class Library:
    """The model files the page may open, and the one loaded now (one at a time)."""

    def __init__(self, paths: Sequence[str], fp16: bool = False, session: Optional[Session] = None,
                 allow_add: bool = True):
        # allow_add: the page may add model files by path (/api/add). serve() allows it only when
        # listening on this computer alone, so a page open to a network cannot probe the disk.
        self.allow_add = allow_add
        seen, self.paths = set(), []
        for p in paths:
            p = str(Path(p).resolve())
            if p not in seen:
                seen.add(p)
                self.paths.append(p)
        if session is not None and str(Path(session.path).resolve()) not in seen:
            self.paths.insert(0, str(Path(session.path).resolve()))
        if not self.paths:
            raise ValueError("no model files")
        self.fp16 = fp16
        self.lock = threading.Lock()
        self._meta: Dict[str, Dict[str, str]] = {}
        self.session = session or Session(self.paths[0], fp16=fp16)

    @staticmethod
    def find(args: Sequence[str]) -> List[str]:
        """Model files from files and folders (a folder: its .safetensors files that hold a model config,
        not, e.g., reference activations)."""
        import tfi
        out: List[str] = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                for f in sorted(p.glob("*.safetensors")):
                    try:
                        if "config" in tfi.read_metadata(str(f)):
                            out.append(str(f))
                    except (RuntimeError, ValueError):
                        pass  # not readable as safetensors: not a model
            elif p.is_file():
                try:
                    meta = tfi.read_metadata(str(p))
                except (RuntimeError, ValueError) as e:
                    raise ValueError(f"{a} is not a safetensors file ({e})") from None
                if "config" not in meta:
                    raise ValueError(f"{a} holds no model (no config in its header)")
                out.append(str(p))
            else:
                raise ValueError(f"no such file or folder: {a}")
        if not out:
            raise ValueError("no model files found in " + ", ".join(args))
        return out

    def _describe(self, i: int, path: str) -> dict:
        if path not in self._meta:
            try:
                self._meta[path] = self.session.tfi.read_metadata(path)
            except (RuntimeError, ValueError):
                self._meta[path] = {}
        meta = self._meta[path]
        try:
            cfg = json.loads(meta.get("config", "{}"))
        except ValueError:
            cfg = {}
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = None
        return {"index": i, "name": Path(path).name, "folder": str(Path(path).parent), "size": size,
                "source": meta.get("source"), "model_type": meta.get("hf_model_type") or ("indranet" if cfg else None),
                "n_layers": cfg.get("n_layers"), "d_model": cfg.get("d_model"),
                "current": path == str(Path(self.session.path).resolve())}

    def models(self, body: dict) -> dict:
        return {"models": [self._describe(i, p) for i, p in enumerate(self.paths)], "can_add": self.allow_add}

    def select(self, body: dict) -> dict:
        i = body.get("index")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(self.paths):
            raise BadRequest(f"index must be in [0, {len(self.paths)})")
        with self.lock:
            if self.paths[i] != str(Path(self.session.path).resolve()):
                self.session = Session(self.paths[i], fp16=self.fp16)  # the old one is freed once its requests end
            return self.session.info({})

    def add(self, body: dict) -> dict:
        """Offer more models: a file, or a folder's model files, given by path on this computer."""
        if not self.allow_add:
            raise BadRequest("adding models from the page is off when it is served to other computers; "
                             "give them on the command line")
        path = body.get("path")
        if not isinstance(path, str) or not path.strip():
            raise BadRequest("give the path of a model file or folder")
        path = str(Path(path.strip()).expanduser())
        found = self.find([path])
        with self.lock:
            added = [str(Path(p).resolve()) for p in found if str(Path(p).resolve()) not in self.paths]
            self.paths.extend(added)
        out = self.models({})
        out["added"] = [self.paths.index(p) for p in added]
        return out

    def route(self, name: str) -> Optional[Callable[[dict], Any]]:
        if name == "models":
            return self.models
        if name == "select":
            return self.select
        if name == "add":
            return self.add
        return self.session.routes().get(name)


def make_handler(target, quiet: bool = True):
    """A request handler for a Library, or for a single Session (a library of one)."""
    library = target if isinstance(target, Library) else Library([], session=target)

    class Handler(BaseHTTPRequestHandler):
        server_version = "indranet-ui"

        def log_message(self, fmt, *args):  # noqa: N802 - http.server's name
            if not quiet:
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str, extra: Sequence[Tuple[str, str]] = ()):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any):
            self._send(status, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
            f = (STATIC / name).resolve()
            if STATIC not in f.parents or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            self._send(200, f.read_bytes(), ctype)

        def do_POST(self):  # noqa: N802
            fn = library.route(self.path[5:]) if self.path.startswith("/api/") else None
            if fn is None:
                return self._json(404, {"error": f"no API {self.path}"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("the request body must be a JSON object")
                out = fn(body)
            except (BadRequest, ValueError, IndexError, RuntimeError, json.JSONDecodeError) as e:
                if not quiet and not isinstance(e, BadRequest):
                    traceback.print_exc()
                return self._json(400, {"error": str(e)})
            except Exception as e:  # pragma: no cover - a bug: say so instead of hanging the page
                traceback.print_exc(file=sys.stderr)
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            if isinstance(out, tuple):  # (float32 array, shape)
                arr, shape = out
                return self._send(200, np.ascontiguousarray(arr, dtype="<f4").tobytes(), "application/octet-stream",
                                  [("X-Shape", ",".join(map(str, shape)))])
            self._json(200, out)

    return Handler


def serve(paths, host: str = "127.0.0.1", port: int = 8765, fp16: bool = False, quiet: bool = True,
          ready: Optional[Callable[[str], None]] = None) -> None:
    """Serve the page for the model files (and folders of them) in `paths`; the first is loaded."""
    loopback = host in ("127.0.0.1", "localhost", "::1")
    library = Library(Library.find([paths] if isinstance(paths, str) else paths), fp16=fp16, allow_add=loopback)
    httpd = ThreadingHTTPServer((host, port), make_handler(library, quiet))
    httpd.daemon_threads = True
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{httpd.server_address[1]}/"
    if ready:
        ready(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
