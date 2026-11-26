"""The Python bindings (python/tfi): built with `make python`, checked against PyTorch and the engine's own
invariants. Skipped when CMake or nanobind is missing."""
import importlib.util
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from indranet import DecoderLM, ModelConfig
from indranet.export import export_model

ROOT = Path(__file__).resolve().parents[2]
V = 61

pytestmark = pytest.mark.skipif(shutil.which("cmake") is None or importlib.util.find_spec("nanobind") is None,
                                reason="needs CMake and nanobind (pip install nanobind)")


@pytest.fixture(scope="module")
def tfi():
    subprocess.run(["make", "-s", "-C", str(ROOT), "python", f"PYTHON={sys.executable}"], check=True,
                   stdout=subprocess.DEVNULL)
    import tfi
    return tfi


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def exported(tmp_path, name="m", vocab=None, **kw):
    torch.manual_seed(0)
    model = DecoderLM(tiny(**kw)).eval()
    path = tmp_path / f"{name}.safetensors"
    export_model(model, path, vocab=vocab)
    return model, path


TOKENS = [5, 9, 13, 2, 7, 5, 9, 13, 2]
CORRUPT = [5, 9, 40, 2, 7, 5, 9, 13, 2]


@pytest.mark.parametrize("kw", [dict(), dict(norm="layernorm", positional="learned", mlp="mlp", bias=True),
                                dict(n_kv_heads=2, qk_norm=True, layer_overrides={1: {"n_heads": 2}})],
                         ids=["llama", "gpt2", "mixed"])
def test_forward_and_cache_match_pytorch(tfi, tmp_path, kw):
    model, path = exported(tmp_path, **kw)
    m = tfi.Model.load(str(path))
    assert m.n_layers == 2 and m.vocab_size == V and m.d_model == 32
    assert m.layer_heads == [4, 2 if "layer_overrides" in kw else 4]
    assert sorted(m.hook_names()) == sorted(model.hook_names())
    with torch.no_grad():
        out, ref = model.run_with_cache(torch.tensor([TOKENS]))
    logits, cache = m.run_with_cache(TOKENS)
    assert logits.dtype == np.float32 and logits.shape == (len(TOKENS), V)
    np.testing.assert_allclose(logits, out.logits[0].numpy(), atol=1e-4)
    np.testing.assert_allclose(m.forward(TOKENS), logits, atol=0)
    assert set(cache) == set(ref)
    for name, act in ref.items():  # PyTorch's have a batch dimension (but hook_pos_embed's)
        a = act.numpy()
        np.testing.assert_allclose(cache[name], a[0] if a.ndim == cache[name].ndim + 1 else a, atol=1e-4, err_msg=name)
    some = ["blocks.1.attn.hook_pattern", "hook_final"]
    _, part = m.run_with_cache(TOKENS, names=some)
    assert sorted(part) == sorted(some)
    with pytest.raises(ValueError, match="no hook point named"):
        m.run_with_cache(TOKENS, names=["blocks.9.hook_resid_pre"])


def test_hooks(tfi, tmp_path):
    model, path = exported(tmp_path)
    m = tfi.Model.load(str(path))
    _, cache = m.run_with_cache(TOKENS)

    def zero_head(z, name):  # in place
        z[:, 1, :] = 0

    def zero_head_returned(z, name):  # by returning a new array (float64 is converted)
        z = z.astype(np.float64)
        z[:, 1, :] = 0
        return z

    def torch_zero(z, name):
        z = z.clone()
        z[:, :, 1, :] = 0
        return z

    with torch.no_grad(), model.hooks({"blocks.0.attn.hook_z": torch_zero}):
        want = model(torch.tensor([TOKENS])).logits[0].numpy()
    for fn in (zero_head, zero_head_returned):
        got = m.forward(TOKENS, hooks={"blocks.0.attn.hook_z": fn})
        np.testing.assert_allclose(got, want, atol=1e-4)
    # The hook sees the activation as it is at that point, and its name.
    seen = {}
    m.forward(TOKENS, hooks={"blocks.1.hook_resid_pre": lambda a, n: seen.update({n: a.copy()})})
    np.testing.assert_allclose(seen["blocks.1.hook_resid_pre"], cache["blocks.1.hook_resid_pre"], atol=0)
    # run_with_cache stores activations after the hooks ran.
    _, hooked = m.run_with_cache(TOKENS, names=["blocks.0.attn.hook_z"], hooks={"blocks.0.attn.hook_z": zero_head})
    assert not hooked["blocks.0.attn.hook_z"][:, 1].any()
    # forward_from with block 1's cached input reproduces forward.
    np.testing.assert_allclose(m.forward_from(1, cache["blocks.1.hook_resid_pre"]), m.forward(TOKENS), atol=1e-5)
    np.testing.assert_allclose(m.embed(TOKENS), cache["blocks.0.hook_resid_pre"], atol=0)
    final = m.final_norm(cache["blocks.1.hook_resid_post"])
    np.testing.assert_allclose(m.logits_from_final(final), m.forward(TOKENS), atol=1e-5)

    with pytest.raises(ValueError, match="boom"):
        m.forward(TOKENS, hooks={"hook_embed": lambda a, n: (_ for _ in ()).throw(ValueError("boom"))})
    with pytest.raises(ValueError, match="returned shape"):
        m.forward(TOKENS, hooks={"hook_embed": lambda a, n: a[:2]})
    with pytest.raises(ValueError, match="no hook point named"):
        m.forward(TOKENS, hooks={"hook_nope": zero_head})
    with pytest.raises(ValueError, match="not callable"):
        m.forward(TOKENS, hooks={"hook_embed": 3})


def test_analyses(tfi, tmp_path):
    model, path = exported(tmp_path)
    m = tfi.Model.load(str(path))
    logits = m.forward(TOKENS)
    T = len(TOKENS)

    # Logit lens: lens_grid at each position agrees with logit_lens; its last stage is the model's output.
    grid = tfi.lens_grid(m, TOKENS, k=4)
    assert grid["stages"] == ["embed", "L0.attn", "L0.mlp", "L1.attn", "L1.mlp"]
    assert grid["top_tokens"].shape == (5, T, 4) and grid["top_tokens"].dtype == np.int64
    for pos in (0, 4, T - 1):
        entries = tfi.logit_lens(m, TOKENS, pos=pos, target=7, k=4)
        for s, e in enumerate(entries):
            assert e["stage"] == grid["stages"][s]
            assert list(grid["top_tokens"][s, pos]) == e["top_tokens"]
            np.testing.assert_allclose(grid["top_probs"][s, pos], e["top_probs"], atol=1e-6)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    np.testing.assert_allclose(grid["next_prob"][-1, :-1], p[np.arange(T - 1), TOKENS[1:]], atol=1e-5)
    assert np.isnan(grid["next_prob"][:, -1]).all() and (grid["next_rank"][:, -1] == -1).all()
    assert tfi.logit_lens(m, TOKENS, pos=-1)[-1]["top_tokens"][0] == int(logits[-1].argmax())

    # DLA sums to the logit difference.
    d = tfi.direct_logit_attribution(m, TOKENS, pos=-1, target=7, other=3)
    assert len(d["labels"]) == len(d["values"])
    assert d["actual"] == pytest.approx(float(logits[-1, 7] - logits[-1, 3]), abs=1e-4)
    assert d["sum"] == pytest.approx(d["actual"], abs=1e-4)
    assert tfi.logit_metric(logits, T - 1, 7, 3) == pytest.approx(d["actual"], abs=1e-5)

    # Patching the last residual stream at the measured position restores the clean metric exactly.
    r = tfi.patch(m, TOKENS, CORRUPT, "resid_post", pos=-1, target=7, other=3)
    assert r["values"].shape == (2, T) and r["row_labels"] == ["L0", "L1"] and r["normalized"]
    assert r["values"][1, -1] == pytest.approx(1.0, abs=1e-4)
    heads = tfi.patch(m, TOKENS, CORRUPT, "heads", pos=-1, target=7, other=3, method="mixed", verify=3)
    assert heads["values"].shape == (2, 4) and heads["exact"].sum() == 3 and heads["estimate"].shape == (2, 4)
    atp = tfi.patch(m, TOKENS, CORRUPT, "resid_pre", pos=-1, target=7, other=3, method="attribution", steps=2)
    assert atp["effect"].shape == (2, T)
    with pytest.raises(ValueError, match="method must be"):
        tfi.patch(m, TOKENS, CORRUPT, "heads", pos=-1, target=7, method="fast")

    # Zero-ablating a head is the loss of the model with that head's hook_z zeroed.
    abl = tfi.head_ablation(m, TOKENS, mode="zero")
    assert abl.shape == (2, 4)
    base = tfi.next_token_loss(logits, TOKENS)

    def zero(z, name):
        z[:, 2, :] = 0
    ablated = tfi.next_token_loss(m.forward(TOKENS, hooks={"blocks.1.attn.hook_z": zero}), TOKENS)
    assert abl[1, 2] == pytest.approx(ablated - base, abs=1e-4)

    metric, grads = tfi.metric_gradients(m, TOKENS, pos=-1, target=7, other=3, names=["blocks.1.hook_resid_post"])
    assert metric == pytest.approx(d["actual"], abs=1e-4) and grads["blocks.1.hook_resid_post"].shape == (T, 32)
    assert not grads["blocks.1.hook_resid_post"][:-1].any()  # only the measured position matters there

    scores = tfi.head_scores(m, segment_len=6, n_samples=2)
    assert scores["induction"].shape == (2, 4) and len(scores["top_two_hop"]) > 0
    comp = tfi.composition_scores(m)
    assert len(comp["labels"]) == 8 and comp["q"].shape == (8, 8)
    assert tfi.ov_copying_scores(m, max_tokens=32).shape == (2, 4)
    w = tfi.head_weights(m, 0, 1)
    assert w["W_Q"].shape == (8, 32) and w["W_O"].shape == (32, 8)
    state = model.state_dict()
    for name in ("blocks.0.attn.o_proj.weight", "blocks.1.ln1.weight"):
        np.testing.assert_allclose(m.weight(name), state[name].numpy(), atol=0)
    assert "embed.weight" in m.weight_names()
    with pytest.raises(IndexError):
        tfi.logit_lens(m, TOKENS, pos=T)


def test_vocab_tokenizer_and_settings(tfi, tmp_path):
    vocab = [chr(ord("a") + i % 26) + (str(i // 26) if i >= 26 else "") for i in range(V)]
    _, path = exported(tmp_path, vocab=vocab)
    m = tfi.Model.load(str(path), fp16=True)
    assert m.fp16 and not m.has_tokenizer and m.vocab == vocab
    assert m.tokenize("abc") == [0, 1, 2] and m.token_str(2) == "c"
    meta = tfi.read_metadata(str(path))
    assert meta["format"] == "indranet" and "config" in meta
    tok = tfi.Tokenizer.from_json('{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}}')
    assert tok.encode("abab") == [2, 2]
    before = tfi.device()
    try:
        tfi.set_device("cpu")
        assert tfi.device() == "cpu"
        with pytest.raises(ValueError, match="device must be"):
            tfi.set_device("tpu")
    finally:
        tfi.set_device(before)
    threads = tfi.num_threads()
    tfi.set_num_threads(2)
    assert tfi.num_threads() == 2
    tfi.set_num_threads(threads)


def test_releases_the_gil(tfi, tmp_path):
    """Two threads run the engine at once and get the same answers as one after the other."""
    _, path = exported(tmp_path, d_model=64, n_layers=4, max_seq_len=512)
    m = tfi.Model.load(str(path))
    seq = list(np.random.default_rng(0).integers(0, V, 256))
    want = m.forward(seq)
    out = [None, None]

    def work(i):
        out[i] = m.forward(seq)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for o in out:
        np.testing.assert_allclose(o, want, atol=0)


@pytest.mark.parametrize("kw", [dict(), dict(norm="layernorm", positional="learned", mlp="mlp", bias=True),
                                dict(n_kv_heads=2, qk_norm=True, positional="alibi", sliding_window=4,
                                     layer_overrides={1: {"n_heads": 2}})],
                         ids=["llama", "gpt2", "mixed"])
def test_generate_matches_pytorch(tfi, tmp_path, kw):
    """Greedy generation with the KV cache = PyTorch's argmax over a full forward pass at every step, and
    indranet's own generate (its KV cache)."""
    model, path = exported(tmp_path, **kw)
    m = tfi.Model.load(str(path))
    got = tfi.generate(m, TOKENS, max_new_tokens=12)
    seq = list(TOKENS)
    with torch.no_grad():
        for _ in range(12):
            seq.append(int(model(torch.tensor([seq])).logits[0, -1].argmax()))
        ref = model.generate(torch.tensor([TOKENS]), 12, temperature=0)
    assert got == seq[len(TOKENS):]
    assert got == ref[0, len(TOKENS):].tolist()


def test_kv_cache(tfi, tmp_path):
    _, path = exported(tmp_path, n_kv_heads=2)
    m = tfi.Model.load(str(path))
    seq = TOKENS + [3, 8, 1]
    want, acts = m.run_with_cache(seq)
    cache = tfi.KVCache()
    parts = [m.forward_cached(seq[:5], cache)] + [m.forward_cached([t], cache) for t in seq[5:]]
    np.testing.assert_allclose(np.concatenate(parts), want, rtol=1e-4, atol=1e-5)
    assert len(cache) == cache.size == len(seq)
    np.testing.assert_allclose(cache.keys(1), acts["blocks.1.attn.hook_k"], rtol=1e-4, atol=1e-5)
    assert cache.values(0).shape == (2, len(seq), 8)
    # A copy continues on its own; truncate rewinds.
    branch = cache.copy()
    branch.truncate(5)
    np.testing.assert_allclose(m.forward_cached(seq[5:], branch, last_only=True), want[-1:], rtol=1e-4, atol=1e-5)
    assert len(cache) == len(seq) and len(branch) == len(seq)
    # Hooks see the new rows; the pattern covers every key.
    seen = {}

    def keep(a, name):
        seen[name] = a.copy()

    c = tfi.KVCache()
    m.forward_cached(seq[:9], c)
    m.forward_cached(seq[9:], c, hooks={"blocks.0.attn.hook_pattern": keep, "blocks.1.hook_resid_post": keep})
    np.testing.assert_allclose(seen["blocks.0.attn.hook_pattern"], acts["blocks.0.attn.hook_pattern"][:, 9:, :],
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(seen["blocks.1.hook_resid_post"], acts["blocks.1.hook_resid_post"][9:], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError):
        m.forward_cached([], c)


def test_kv_cache_clamped(tfi, tmp_path):
    """An fp16 cache counts the keys and values it cut off at fp16's limit; generate warns about them."""
    _, path = exported(tmp_path)
    m16, m32 = tfi.Model.load(str(path), fp16=True), tfi.Model.load(str(path))

    def huge(a, name):
        return a * 1e6

    hooks = {"blocks.0.attn.hook_k": huge}
    c16, c32, plain = tfi.KVCache(), tfi.KVCache(), tfi.KVCache()
    m16.forward_cached(TOKENS, c16, hooks=hooks)
    m32.forward_cached(TOKENS, c32, hooks=hooks)
    m16.forward_cached(TOKENS, plain)
    assert c16.clamped > 0 and c16.clamped == int((np.abs(c16.keys(0)) == 65504).sum())
    assert c32.clamped == 0 and plain.clamped == 0
    with pytest.warns(RuntimeWarning, match="cut off at"):
        tfi.generate(m16, TOKENS, max_new_tokens=2, hooks=hooks)
    with pytest.warns(RuntimeWarning, match="cut off at"):
        tfi.generate_batch(m16, [TOKENS, TOKENS[:3]], max_new_tokens=2, hooks=[None, hooks])


def test_prompt_cache(tfi, tmp_path):
    """A PromptCache: the next turn reuses the earlier text and gives a fresh call's tokens; editing hooks refused."""
    _, path = exported(tmp_path)
    m = tfi.Model.load(str(path))
    pc = tfi.PromptCache()
    out1 = tfi.generate(m, TOKENS, max_new_tokens=6, prompt_cache=pc)
    assert pc.reused == 0 and pc.tokens == list(TOKENS) + out1[:-1] and len(pc) == len(pc.tokens)
    turn = list(TOKENS) + out1 + [5, 9, 2]
    out2 = tfi.generate(m, turn, max_new_tokens=6, prompt_cache=pc)
    assert pc.reused == len(TOKENS) + len(out1) - 1
    assert out2 == tfi.generate(m, turn, max_new_tokens=6)
    with pytest.raises(ValueError):
        tfi.generate(m, turn, prompt_cache=pc, hooks={"blocks.0.hook_attn_out": lambda a, name: a * 0})
    tfi.generate(m, turn, prompt_cache=pc, captures={"blocks.0.hook_resid_pre": lambda a, name: None})
    assert pc.reused == len(turn) - 1
    pc.clear()
    assert len(pc) == 0 and pc.tokens == []


def test_generate_options(tfi, tmp_path):
    vocab = [chr(ord("a") + i % 26) + (str(i // 26) if i >= 26 else "") for i in range(V)]
    _, path = exported(tmp_path, vocab=vocab)
    m = tfi.Model.load(str(path))
    assert m.eos_tokens == []
    greedy = tfi.generate(m, TOKENS, max_new_tokens=10)
    assert len(greedy) == 10
    assert tfi.generate(m, "abc", max_new_tokens=3) == tfi.generate(m, [0, 1, 2], max_new_tokens=3)
    assert m.decode([0, 1, 2]) == "abc"
    # Stops: a stop token (returned), the model's eos tokens, on_token returning False.
    assert tfi.generate(m, TOKENS, max_new_tokens=10, stop_tokens=[greedy[3]])[-1] == greedy[3]
    m.eos_tokens = [greedy[2]]
    assert len(tfi.generate(m, TOKENS, max_new_tokens=10)) <= 3
    assert tfi.generate(m, TOKENS, max_new_tokens=10, stop_at_eos=False) == greedy
    m.eos_tokens = []
    seen = []
    assert tfi.generate(m, TOKENS, max_new_tokens=10, on_token=lambda t: seen.append(t) or len(seen) < 4) == greedy[:4]
    assert seen == greedy[:4]
    # Sampling: reproducible from its seed; top_k=1 is greedy.
    a = tfi.generate(m, TOKENS, max_new_tokens=10, temperature=1.0, seed=7)
    assert a == tfi.generate(m, TOKENS, max_new_tokens=10, temperature=1.0, seed=7)
    assert tfi.generate(m, TOKENS, max_new_tokens=10, temperature=1.0, top_k=1) == greedy
    # Hooks apply at every step.
    def zero(a, name):
        a[:] = 0
    hooked = tfi.generate(m, TOKENS, max_new_tokens=6, hooks={"blocks.1.hook_attn_out": zero})
    seq = list(TOKENS)
    for _ in range(6):
        seq.append(int(m.forward(seq, hooks={"blocks.1.hook_attn_out": zero})[-1].argmax()))
    assert hooked == seq[len(TOKENS):]
    with pytest.raises(ValueError, match="temperature"):
        tfi.generate(m, TOKENS, temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        tfi.generate(m, TOKENS, top_p=0.0)

    def boom(t):
        raise RuntimeError("stop here")
    with pytest.raises(RuntimeError, match="stop here"):
        tfi.generate(m, TOKENS, on_token=boom)


def _hf_decoder(style):
    """The decoder models of each kind ship: byte-level, or Llama 2's SentencePiece decoder (without the
    final Strip where the encoder prepends no ▁, as Gemma's). tokenizers' Metaspace decoder is left out:
    it drops every ▁ of the first token, not just the one the encoder added."""
    from tokenizers import decoders
    if style in ("llama2_legacy", "llama2_metaspace", "metaspace_split", "gemma"):
        steps = [decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()]
        return decoders.Sequence(steps + ([] if style == "gemma" else [decoders.Strip(" ", 1, 0)]))
    return decoders.ByteLevel()


def test_tokenizer_decode_matches_tokenizers(tfi):
    """Tokenizer.decode = tokenizers' decode with the decoder each kind of model ships, for every kind of
    tokenizer the engine implements, special tokens kept and skipped."""
    pytest.importorskip("tokenizers")
    import json
    from test_hf import TOKENIZER_STYLES, TOKENIZER_TEXTS, MULTILINE_TEXTS, _tokenizer
    for style in TOKENIZER_STYLES:
        hf, d = _tokenizer(style)
        hf.decoder = _hf_decoder(style)
        stored = {k: d.get(k) for k in ("normalizer", "pre_tokenizer", "model", "post_processor", "added_tokens")}
        tok = tfi.Tokenizer.from_json(json.dumps(stored))
        for text in TOKENIZER_TEXTS + MULTILINE_TEXTS:
            for special in (True, False):
                ids = hf.encode(text, add_special_tokens=special).ids
                for skip in (False, True):
                    assert tok.decode(ids, skip) == hf.decode(ids, skip_special_tokens=skip), (style, text, special, skip)


def test_generate_trace_and_captures(tfi, tmp_path):
    """generate(trace=True): per new token its probability (= the full pass's), top tokens, logit lens, DLA
    adding up to its logit, and each head's most attended positions; captures read without editing."""
    model, path = exported(tmp_path, n_kv_heads=2)
    m = tfi.Model.load(str(path))
    new, steps = tfi.generate(m, TOKENS, max_new_tokens=4, trace=True, trace_top_k=3, trace_attention=2)
    assert new == tfi.generate(m, TOKENS, max_new_tokens=4) and len(steps) == 4
    seq = list(TOKENS)
    for tok, st in zip(new, steps):
        assert st["token"] == tok and st["position"] == len(seq) - 1
        logits = m.forward(seq)[-1].astype(np.float64)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        assert st["prob"] == pytest.approx(p[tok], rel=1e-4) and st["logprob"] == pytest.approx(np.log(p[tok]), abs=1e-3)
        assert len(st["top_tokens"]) == 3 and st["top_tokens"][0] == int(p.argmax())
        lens = st["lens"]
        assert lens["stages"][0] == "embed" and lens["stages"][-1] == "L1.mlp" and len(lens["prob"]) == 5
        assert lens["rank"][-1] == 0 and lens["top"][-1] == tok  # the last stage is the model's own (greedy) choice
        dla = st["dla"]
        assert dla["sum"] == pytest.approx(dla["actual"], rel=1e-3, abs=1e-3) and len(dla["labels"]) == len(dla["values"])
        assert sorted(st["attention"]) == sorted(f"L{l}H{h}" for l in range(2) for h in range(4))
        assert all(len(a["positions"]) == 2 and a["weights"][0] >= a["weights"][1] for a in st["attention"].values())
        seq.append(tok)
    # captures: called on every pass, the prompt's rows then one per new token; what they return is ignored
    shapes = []
    out = tfi.generate(m, TOKENS, max_new_tokens=3,
                       captures={"blocks.1.hook_resid_post": lambda a, n: shapes.append(a.shape) or np.zeros_like(a)})
    assert out == new[:3] and shapes == [(len(TOKENS), 32), (1, 32), (1, 32)]
    want, acts = m.run_with_cache(TOKENS, names=["blocks.0.hook_mlp_out"])
    got = {}
    logits = m.forward(TOKENS, captures={"blocks.0.hook_mlp_out": lambda a, n: got.setdefault(n, a.copy())})
    np.testing.assert_array_equal(logits, want)
    np.testing.assert_array_equal(got["blocks.0.hook_mlp_out"], acts["blocks.0.hook_mlp_out"])
    with pytest.raises(ValueError, match="post-norm"):
        _, post = exported(tmp_path, name="post", norm_position="post", norm="layernorm")
        tfi.generate(tfi.Model.load(str(post)), TOKENS, max_new_tokens=2, trace=True)


def test_generate_batch(tfi, tmp_path):
    """generate_batch: sequence i = generate() with seed + i; hooks per sequence or for all; captures see each
    sequence's own rows; traces per sequence."""
    model, path = exported(tmp_path, n_kv_heads=2)
    m = tfi.Model.load(str(path))
    prompts = [TOKENS, [3, 8, 1], [7] * 6]
    assert tfi.generate_batch(m, prompts, max_new_tokens=5) == [tfi.generate(m, p, max_new_tokens=5) for p in prompts]
    sampled = tfi.generate_batch(m, prompts, max_new_tokens=5, temperature=1.0, seed=4)
    assert sampled == [tfi.generate(m, p, max_new_tokens=5, temperature=1.0, seed=4 + i) for i, p in enumerate(prompts)]

    def zero(a, name):
        a[:] = 0
    hooked = tfi.generate_batch(m, prompts, max_new_tokens=5, hooks=[None, {"blocks.1.hook_attn_out": zero}, None])
    assert hooked[0] == tfi.generate(m, prompts[0], max_new_tokens=5)
    assert hooked[1] == tfi.generate(m, prompts[1], max_new_tokens=5, hooks={"blocks.1.hook_attn_out": zero})
    assert tfi.generate_batch(m, prompts, max_new_tokens=5, hooks={"blocks.1.hook_attn_out": zero}) == \
        [tfi.generate(m, p, max_new_tokens=5, hooks={"blocks.1.hook_attn_out": zero}) for p in prompts]
    rows = {i: [] for i in range(3)}
    caps = [{"blocks.0.hook_resid_post": (lambda i: lambda a, n: rows[i].append(a.shape[0]))(i)} for i in range(3)]
    tfi.generate_batch(m, prompts, max_new_tokens=3, captures=caps)
    assert rows == {0: [len(TOKENS), 1, 1], 1: [3, 1, 1], 2: [6, 1, 1]}  # each its prompt, then one row per step
    new, traces = tfi.generate_batch(m, prompts, max_new_tokens=4, trace=True)
    assert len(traces) == 3 and all(len(t) == 4 for t in traces)
    for p, toks, t in zip(prompts, new, traces):
        _, one = tfi.generate(m, p, max_new_tokens=4, trace=True)
        assert [s["token"] for s in t] == toks == [s["token"] for s in one]
        assert [s["prob"] for s in t] == pytest.approx([s["prob"] for s in one], rel=1e-4)
    with pytest.raises(ValueError, match="one per prompt"):
        tfi.generate_batch(m, prompts, hooks=[None])


def test_generate_speculative(tfi, tmp_path):
    """speculate=K: greedy output = plain greedy whoever proposes (prompt lookup, another model, the model
    itself), with hooks and the trace too; stats counts the steps; sampling keeps the model's distribution."""
    _, path = exported(tmp_path, n_kv_heads=2)
    m = tfi.Model.load(str(path))
    torch.manual_seed(1)
    other_path = tmp_path / "draft.safetensors"
    export_model(DecoderLM(tiny(d_model=16, n_layers=1)).eval(), other_path)
    draft = tfi.Model.load(str(other_path))
    plain = tfi.generate(m, TOKENS, max_new_tokens=12)
    for k in (1, 4):
        for d in (None, draft, m):
            stats = {}
            assert tfi.generate(m, TOKENS, max_new_tokens=12, speculate=k, draft=d, stats=stats) == plain
            assert stats["steps"] + stats["accepted"] == 12 and stats["accepted"] <= stats["drafted"]
            if d is m:
                assert stats["accepted"] == stats["drafted"]
    assert tfi.generate(m, TOKENS, max_new_tokens=12, speculate=3, ngram=1) == plain

    def half(a, name):
        a *= 0.5
    hooks = {"blocks.0.hook_attn_out": half}
    assert tfi.generate(m, TOKENS, max_new_tokens=8, hooks=hooks, speculate=3, draft=m) == \
        tfi.generate(m, TOKENS, max_new_tokens=8, hooks=hooks)
    new, steps = tfi.generate(m, TOKENS, max_new_tokens=6, trace=True, speculate=3, draft=draft)
    new0, steps0 = tfi.generate(m, TOKENS, max_new_tokens=6, trace=True)
    assert new == new0 and [s["position"] for s in steps] == [s["position"] for s in steps0]
    assert [s["prob"] for s in steps] == pytest.approx([s["prob"] for s in steps0], rel=1e-4)
    for s, s0 in zip(steps, steps0):
        np.testing.assert_allclose(s["lens"]["prob"], s0["lens"]["prob"], rtol=1e-4, atol=1e-6)
    # Hooks that keep state or pick a row: the same text, calls and state as without speculation.
    def counting():
        seen = []

        def edit(a, name):
            seen.append(a.shape)
            if len(seen) % 3 == 0:
                a[-1] *= -1
        return {"blocks.0.hook_attn_out": edit}, seen

    def carrying():
        kept = []

        def use(a, name):
            kept.append(a.shape)
            if len(kept) > 1:
                a += 0.5 * np.tanh(saved[0])

        saved = [None]

        def save(a, name):
            saved[0] = a[-1].copy()
        return {"blocks.0.hook_attn_out": use, "blocks.1.hook_resid_post": save}, kept
    for make in (counting, carrying):
        hooks, seen = make()
        want = tfi.generate(m, TOKENS, max_new_tokens=10, hooks=hooks)
        calls = list(seen)
        for d in (None, draft, m):
            hooks, seen = make()
            stats = {}
            assert tfi.generate(m, TOKENS, max_new_tokens=10, hooks=hooks, speculate=4, draft=d, stats=stats) == want
            assert seen == calls and calls[0][0] == len(TOKENS) and "reruns" in stats
    # Sampling: the second token, the first one a step can propose, follows the model's distribution.
    softmax = lambda logits: torch.softmax(torch.tensor(logits, dtype=torch.float64) / 2.0, 0).numpy()
    first = softmax(m.forward(TOKENS)[-1])
    p = sum(first[a] * softmax(m.forward(TOKENS + [a])[-1]) for a in range(V))
    n = 3000
    for d in (None, draft):
        counts = np.zeros(V)
        for seed in range(n):
            counts[tfi.generate(m, TOKENS, max_new_tokens=3, temperature=2.0, seed=seed, speculate=2, draft=d)[1]] += 1
        assert np.all(np.abs(counts / n - p) <= 5 * np.sqrt(p * (1 - p) / n) + 1e-3)
    with pytest.raises(ValueError, match="draft needs speculate"):
        tfi.generate(m, TOKENS, draft=draft)
    with pytest.raises(ValueError, match="ngram"):
        tfi.generate(m, TOKENS, speculate=2, ngram=0)
