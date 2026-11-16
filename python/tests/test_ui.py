"""The UI server (indranet_ui): every API route on a tiny model, over HTTP. Skipped when the tfi bindings
can't be built (needs CMake and nanobind)."""
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch

from indranet import DecoderLM, ModelConfig
from indranet.export import export_model

ROOT = Path(__file__).resolve().parents[2]
V = 40
VOCAB = [chr(ord("a") + i) for i in range(26)] + [str(i) for i in range(10)] + [" ", ".", ",", "!"]

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None or importlib.util.find_spec("nanobind") is None,
                                reason="needs CMake and nanobind (pip install nanobind)")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    subprocess.run(["make", "-s", "-C", str(ROOT), "python", f"PYTHON={sys.executable}"], check=True,
                   stdout=subprocess.DEVNULL)
    from indranet_ui.server import Session, make_handler

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ui") / "tiny.safetensors"
    export_model(DecoderLM(ModelConfig(vocab_size=V, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)), path, vocab=VOCAB)
    session = Session(str(path))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", session
    httpd.shutdown()
    httpd.server_close()


def post(base, name, body=None, raw=False):
    req = urllib.request.Request(f"{base}/api/{name}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return (r.status, data, dict(r.headers)) if raw else (r.status, json.loads(data))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_files(server):
    base, _ = server
    for path, kind in [("/", "text/html"), ("/app.js", "javascript"), ("/style.css", "text/css")]:
        with urllib.request.urlopen(base + path) as r:
            assert r.status == 200 and kind in r.headers["Content-Type"]
    for bad in ("/nope.js", "/../server.py", "/%2e%2e/server.py"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + bad)
        assert e.value.code == 404


def test_info_tokenize_run(server):
    base, session = server
    status, info = post(base, "info")
    assert status == 200 and info["n_layers"] == 2 and info["layer_heads"] == [4, 4] and info["has_vocab"]
    assert info["config"]["d_model"] == 32 and not info["has_tokenizer"]
    status, tok = post(base, "tokenize", {"text": "hello world"})
    assert status == 200 and tok["texts"] == list("hello world")
    tokens = tok["tokens"]
    status, run = post(base, "run", {"tokens": tokens, "k": 3})
    assert status == 200 and len(run["top"]) == len(tokens) and len(run["top"][0]) == 3
    logits = session.model.forward(tokens)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    assert run["top"][2][0][0] == int(p[2].argmax())
    assert run["next_prob"][0] == pytest.approx(float(p[0, tokens[1]]), rel=1e-5) and run["next_prob"][-1] is None
    assert run["loss"] == pytest.approx(float(-np.mean(np.log(p[np.arange(len(tokens) - 1), tokens[1:]]))), rel=1e-5)
    assert post(base, "texts", {"tokens": [7, 4]})[1]["texts"] == ["h", "e"]


def test_attention_is_binary(server):
    base, session = server
    tokens = [7, 4, 11, 11, 14]
    status, data, headers = post(base, "attention", {"tokens": tokens, "layer": 1}, raw=True)
    assert status == 200 and headers["X-Shape"] == "4,5,5"
    got = np.frombuffer(data, dtype="<f4").reshape(4, 5, 5)
    _, cache = session.model.run_with_cache(tokens, names=["blocks.1.attn.hook_pattern"])
    np.testing.assert_array_equal(got, cache["blocks.1.attn.hook_pattern"])
    status, data, _ = post(base, "attention", {"tokens": tokens, "layer": 0}, raw=True)  # from the cache
    _, cache = session.model.run_with_cache(tokens, names=["blocks.0.attn.hook_pattern"])
    np.testing.assert_array_equal(np.frombuffer(data, dtype="<f4").reshape(4, 5, 5), cache["blocks.0.attn.hook_pattern"])


def test_analyses(server):
    base, _ = server
    tokens = [7, 4, 11, 11, 14, 36, 22, 14, 17, 11, 3]
    status, lens = post(base, "lens", {"tokens": tokens, "k": 3})
    assert status == 200 and lens["stages"][0] == "embed" and len(lens["top_texts"][0][0]) == 3
    assert lens["next_prob"][0][-1] is None and lens["next_rank"][0][-1] == -1
    status, dla = post(base, "dla", {"tokens": tokens, "pos": -1, "target": "e", "other": "#3"})
    assert status == 200 and dla["target"] == 4 and dla["other"] == 3 and dla["pos"] == len(tokens) - 1
    assert dla["sum"] == pytest.approx(dla["actual"], abs=1e-3)
    status, top = post(base, "dla", {"tokens": tokens, "pos": 2})  # no target: the top prediction
    assert status == 200 and top["other"] == -1
    corrupt = list(tokens)
    corrupt[3] = 12
    status, p = post(base, "patch", {"clean": tokens, "corrupt": corrupt, "pos": -1, "target": "e", "other": "a",
                                     "site": "resid_post"})
    assert status == 200 and len(p["values"]) == 2 and len(p["values"][0]) == len(tokens)
    assert p["values"][1][-1] == pytest.approx(1.0, abs=1e-3)
    status, mixed = post(base, "patch", {"clean": tokens, "corrupt": corrupt, "pos": -1, "target": "e", "site": "heads",
                                         "method": "mixed", "verify": 2})
    assert status == 200 and sum(map(sum, mixed["exact"])) == 2
    status, same = post(base, "patch", {"clean": tokens, "corrupt": tokens, "pos": -1, "target": "e", "site": "heads"})
    assert status == 200 and not same["normalized"] and same["values"] is None
    status, heads = post(base, "heads", {"segment_len": 6, "n_samples": 2})
    assert status == 200 and len(heads["induction"]) == 2 and len(heads["induction"][0]) == 4


@pytest.mark.parametrize("name,body,message", [
    ("run", {"tokens": []}, "non-empty list"),
    ("run", {"tokens": [1, 99]}, "must be in [0, 40)"),
    ("run", {"tokens": list(range(30)) * 3}, "at most 64"),
    ("attention", {"tokens": [1, 2], "layer": 2}, "layer must be"),
    ("dla", {"tokens": [1, 2], "pos": 5, "target": "a"}, "pos must be"),
    ("dla", {"tokens": [1, 2], "target": "ab"}, "is 2 tokens"),
    ("dla", {"tokens": [1, 2], "target": "#77"}, "out of range"),
    ("patch", {"clean": [1, 2], "corrupt": [1], "target": "a"}, "same number of tokens"),
    ("patch", {"clean": [1, 2], "corrupt": [1, 3], "target": "a", "site": "everywhere"}, "unknown site"),
    ("patch", {"clean": [1, 2], "corrupt": [1, 3], "target": "a", "method": "guess"}, "method must be"),
    ("heads", {"segment_len": 40}, "too long"),
    ("tokenize", {"text": ""}, "give some text"),
])
def test_bad_requests(server, name, body, message):
    status, out = post(server[0], name, body)
    assert status == 400 and message in out["error"]


def test_unknown_route_and_bad_json(server):
    base, _ = server
    assert post(base, "shutdown")[0] == 404
    req = urllib.request.Request(f"{base}/api/run", data=b"{not json", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400


def test_model_without_vocabulary(server, tmp_path):
    """Token ids only: text targets are refused with a hint, #ids work, and the page's info says so."""
    from indranet_ui.server import BadRequest, Session

    torch.manual_seed(1)
    path = tmp_path / "ids.safetensors"
    export_model(DecoderLM(ModelConfig(vocab_size=V, d_model=16, n_layers=1, n_heads=2, max_seq_len=32)), path)
    s = Session(str(path))
    assert not s.info({})["has_vocab"] and not s.info({})["has_tokenizer"]
    assert s.token_id("#3", "target") == 3 and s.token_id(5, "target") == 5
    assert s.token_id("", "other", required=False) == -1
    with pytest.raises(BadRequest, match="written #123"):
        s.token_id("a", "target")
    with pytest.raises(BadRequest, match="give token ids"):
        s.tokenize({"text": "abc"})
    assert s.texts({"tokens": [3, 4]})["texts"] == ["3", "4"]


def test_library_lists_and_switches_models(server, tmp_path):
    """A folder of models (other safetensors files skipped): /api/models lists them, /api/select loads one."""
    from indranet.export import save_safetensors
    from indranet_ui.server import Library, make_handler

    folder = tmp_path / "models"
    folder.mkdir()
    for name, layers in (("a.safetensors", 1), ("b.safetensors", 2)):
        torch.manual_seed(layers)
        export_model(DecoderLM(ModelConfig(vocab_size=V, d_model=16, n_layers=layers, n_heads=2, max_seq_len=32)),
                     folder / name, vocab=VOCAB)
    save_safetensors({"x": torch.zeros(2)}, folder / "activations.safetensors", {"note": "not a model"})
    paths = Library.find([str(folder)])
    assert [Path(p).name for p in paths] == ["a.safetensors", "b.safetensors"]
    with pytest.raises(ValueError, match="no such file"):
        Library.find([str(folder / "missing.safetensors")])

    library = Library(paths + paths)  # duplicates collapse
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(library))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, out = post(base, "models")
        assert status == 200 and [m["name"] for m in out["models"]] == ["a.safetensors", "b.safetensors"]
        assert [m["current"] for m in out["models"]] == [True, False]
        assert out["models"][1]["n_layers"] == 2 and out["models"][1]["model_type"] == "indranet"
        assert post(base, "info")[1]["n_layers"] == 1
        status, info = post(base, "select", {"index": 1})
        assert status == 200 and info["n_layers"] == 2 and info["file"] == "b.safetensors"
        assert post(base, "info")[1]["n_layers"] == 2
        assert [m["current"] for m in post(base, "models")[1]["models"]] == [False, True]
        status, run = post(base, "run", {"tokens": [7, 4, 11]})  # the analyses use the new model
        assert status == 200 and len(run["top"]) == 3
        for bad in (2, -1, "1", True):
            status, err = post(base, "select", {"index": bad})
            assert status == 400 and "index must be" in err["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_adding_models_by_path(server, tmp_path):
    """/api/add offers more model files (validated), and is refused when the page is served to other computers."""
    from indranet.export import save_safetensors
    from indranet_ui.server import Library, make_handler

    torch.manual_seed(3)
    first, extra = tmp_path / "first.safetensors", tmp_path / "extra.safetensors"
    for p in (first, extra):
        export_model(DecoderLM(ModelConfig(vocab_size=V, d_model=16, n_layers=1, n_heads=2, max_seq_len=32)), p, vocab=VOCAB)
    save_safetensors({"x": torch.zeros(2)}, tmp_path / "acts.safetensors", {"note": "not a model"})
    (tmp_path / "notes.txt").write_text("hello")
    for allow in (True, False):
        library = Library([str(first)], allow_add=allow)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(library))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            assert post(base, "models")[1]["can_add"] is allow
            status, out = post(base, "add", {"path": str(extra)})
            if not allow:
                assert status == 400 and "off" in out["error"] and len(library.paths) == 1
                continue
            assert status == 200 and out["added"] == [1] and [m["name"] for m in out["models"]] == ["first.safetensors", "extra.safetensors"]
            assert post(base, "add", {"path": str(extra)})[1]["added"] == []  # already offered
            assert post(base, "select", {"index": 1})[1]["file"] == "extra.safetensors"
            for bad, message in [(str(tmp_path / "missing.safetensors"), "no such file"), (str(tmp_path / "acts.safetensors"), "holds no model"),
                                 (str(tmp_path / "notes.txt"), "not a safetensors file"), ("", "give the path")]:
                status, err = post(base, "add", {"path": bad})
                assert status == 400 and message in err["error"], err
        finally:
            httpd.shutdown()
            httpd.server_close()
