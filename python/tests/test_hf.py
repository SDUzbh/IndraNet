"""indranet_hf, the Hugging Face converter.

Unit tests need numpy and torch. The golden tests also need transformers (skipped without it) and a
C++ toolchain: they build tiny random models of each supported family with transformers, convert
them, and check the C++ engine reproduces transformers' logits and every layer's residual stream.
"""
import copy
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from indranet import ModelConfig
from indranet.export import save_safetensors
from indranet_hf import Unsupported, convert, translate_config
from indranet_hf import tensors, vocab
from indranet_hf.families import CONFIG_FIELDS

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ONLY = {"use_sdpa", "dropout", "attn_dropout", "init_std", "scale_residual_init"}

LLAMA2_7B = dict(model_type="llama", hidden_size=4096, intermediate_size=11008, num_attention_heads=32,
                 num_hidden_layers=32, num_key_value_heads=32, max_position_embeddings=4096, rms_norm_eps=1e-5,
                 vocab_size=32000, hidden_act="silu", rope_theta=10000.0, rope_scaling=None, tie_word_embeddings=False)


# --- unit tests --------------------------------------------------------------------------------

def test_bf16_rounding_matches_torch():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(10000) * 10.0 ** torch.randint(-8, 8, (10000,)).float(),
                   torch.tensor([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -(1.0 + 2 ** -8), 0.0, -0.0, 1e-40])])  # ties, subnormal
    ours = tensors.f32_to_bf16(x.numpy())
    assert np.array_equal(ours, x.bfloat16().view(torch.int16).numpy().view(np.uint16))
    assert np.array_equal(tensors.bf16_to_f32(ours), x.bfloat16().float().numpy())


def test_cast_refuses_values_that_dont_fit():
    big = np.array([[70000.0, 1.0]], np.float32)
    assert tensors.cast(big, "F32", "F16") is None
    assert tensors.cast(big, "F32", "BF16") is not None
    assert tensors.cast(np.array([[np.inf]], np.float32), "F32", "BF16") is None
    assert tensors.cast(np.array([[3.4e38]], np.float32), "F32", "BF16") is None  # rounds to bf16's infinity
    assert tensors.cast(np.array([[0.5]], np.float32), "F32", "F16").dtype == np.float16


def test_safetensors_write_and_read(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = tensors.f32_to_bf16(a)
    tensors.write(tmp_path / "t.safetensors", {"a": ("F32", (2, 3), lambda: a), "b": ("BF16", (3, 2), lambda: b.T)},
                  {"k": "v"})
    s = tensors.Shards([tmp_path / "t.safetensors"])
    assert s.names() == ["a", "b"] and s.dtype("b") == "BF16" and s.shape("b") == (3, 2)
    assert np.array_equal(s.get("a"), a)
    assert np.array_equal(tensors.bf16_to_f32(s.get("b")), a.T)


def test_config_fields_match_indranet():
    """The converter writes every engine field of ModelConfig, and nothing else."""
    assert set(CONFIG_FIELDS) == {f.name for f in dataclasses.fields(ModelConfig)} - TRAINING_ONLY


def test_llama2_config():
    c = translate_config(LLAMA2_7B)
    assert (c["d_model"], c["n_layers"], c["n_heads"], c["n_kv_heads"], c["d_head"], c["d_mlp"]) == (4096, 32, 32, 32, 128, 11008)
    assert (c["norm"], c["norm_position"], c["positional"], c["mlp"], c["norm_eps"]) == ("rmsnorm", "pre", "rope", "swiglu", 1e-5)
    assert not c["tie_embeddings"] and not c["layer_overrides"] and c["sliding_window"] is None
    ModelConfig.from_dict(c)  # a config indranet accepts too


def test_gemma2_config():
    hf = dict(model_type="gemma2", hidden_size=3584, intermediate_size=14336, num_attention_heads=16, num_key_value_heads=8,
              head_dim=256, num_hidden_layers=6, vocab_size=256000, rms_norm_eps=1e-6, query_pre_attn_scalar=144,
              sliding_window=4096, attn_logit_softcapping=50.0, final_logit_softcapping=30.0, rope_theta=10000.0)
    c = translate_config(hf)
    assert c["norm_position"] == "sandwich" and c["embed_scale"] and c["tie_embeddings"]
    assert (c["mlp"], c["activation"]) == ("geglu", "gelu_tanh")
    assert c["attn_scale"] == pytest.approx(144 ** -0.5)
    assert (c["attn_logit_softcap"], c["final_logit_softcap"]) == (50.0, 30.0)
    # even layers slide (as transformers' Gemma 2), so half the layers override the common setting
    windows = [c["layer_overrides"].get(str(i), {}).get("sliding_window", c["sliding_window"]) for i in range(6)]
    assert windows == [4096, None, 4096, None, 4096, None]
    ModelConfig.from_dict(c)


def test_gemma3_local_rope():
    hf = dict(model_type="gemma3_text", hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=1,
              head_dim=16, num_hidden_layers=6, vocab_size=97, sliding_window=512, sliding_window_pattern=3,
              rope_theta=1e6, rope_local_base_freq=1e4, query_pre_attn_scalar=16)
    c = translate_config(hf)
    per = [{**c, **c["layer_overrides"].get(str(i), {})} for i in range(6)]
    assert [(p["sliding_window"], p["rope_theta"]) for p in per] == [(512, 1e4), (512, 1e4), (None, 1e6)] * 2
    assert c["qk_norm"] and c["attn_scale"] is None


def test_layer_rules_without_layer_types():
    """Configs saved before transformers wrote layer_types: the sliding layers follow each family's rule."""
    qwen = dict(LLAMA2_7B, model_type="qwen2", num_hidden_layers=4, sliding_window=64, max_window_layers=2)
    windows = lambda c: [c["layer_overrides"].get(str(i), {}).get("sliding_window", c["sliding_window"])
                         for i in range(c["n_layers"])]
    assert windows(translate_config(dict(qwen, use_sliding_window=True))) == [None, None, 64, 64]
    assert windows(translate_config(dict(qwen, use_sliding_window=False))) == [None] * 4
    assert windows(translate_config(dict(qwen, layer_types=["sliding_attention"] * 4, use_sliding_window=True))) == [64] * 4
    assert windows(translate_config(dict(LLAMA2_7B, model_type="mistral", sliding_window=64))) == [64] * 32


def test_newer_rope_parameters():
    hf = dict(LLAMA2_7B, rope_theta=None, rope_parameters={"rope_type": "default", "rope_theta": 500000.0})
    assert translate_config(hf)["rope_theta"] == 500000.0


LLAMA31_SCALING = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                   "original_max_position_embeddings": 8192}


def test_rope_scaling_configs():
    c = translate_config(dict(LLAMA2_7B, rope_theta=500000.0, max_position_embeddings=131072, rope_scaling=LLAMA31_SCALING))
    assert c["rope_scaling"] == {"type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                 "original_max_position_embeddings": 8192}
    ModelConfig.from_dict(c)
    # transformers' defaults made explicit; unset optional values left out
    yarn = translate_config(dict(LLAMA2_7B, rope_scaling={"rope_type": "yarn", "factor": 4.0, "beta_fast": None}))
    assert yarn["rope_scaling"] == {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
    dyn = translate_config(dict(LLAMA2_7B, rope_scaling={"type": "dynamic", "factor": 2.0}))
    assert dyn["rope_scaling"] == {"type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 4096}
    assert translate_config(dict(LLAMA2_7B, rope_scaling={"type": "linear", "factor": 1.0}))["rope_scaling"] is None
    # Phi-3.5: lengths at the top level set the factor; "su" is longrope's older name
    phi = dict(model_type="phi3", hidden_size=3072, intermediate_size=8192, num_attention_heads=32, num_hidden_layers=2,
               vocab_size=32064, max_position_embeddings=131072, original_max_position_embeddings=4096,
               rope_scaling={"type": "su", "short_factor": [1.0] * 48, "long_factor": [2.0] * 48})
    r = translate_config(phi)["rope_scaling"]
    assert (r["type"], r["factor"], r["original_max_position_embeddings"], len(r["short_factor"])) == ("longrope", 32.0, 4096, 48)
    assert "long_factor" not in r  # past original_max_position_embeddings, which the engine refuses


def test_gemma3_scaling_on_global_layers_only():
    hf = dict(model_type="gemma3_text", hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=1,
              head_dim=16, num_hidden_layers=4, vocab_size=97, sliding_window=512, sliding_window_pattern=2,
              rope_theta=1e6, rope_local_base_freq=1e4, query_pre_attn_scalar=16,
              rope_scaling={"rope_type": "linear", "factor": 8.0})
    c = translate_config(hf)
    per = [{**c, **c["layer_overrides"].get(str(i), {})} for i in range(4)]
    assert [p["rope_scaling"] for p in per] == [None, {"type": "linear", "factor": 8.0}] * 2
    newer = dict(hf, rope_scaling=None, rope_theta=None, rope_local_base_freq=None,
                 rope_parameters={"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6},
                                  "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}})
    assert translate_config(newer) == c


def test_unknown_rope_scaling_is_refused():
    with pytest.raises(Unsupported, match="RoPE scaling 'mrope'"):
        translate_config(dict(LLAMA2_7B, rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]}))


def test_unknown_model_type_is_refused():
    with pytest.raises(Unsupported, match="model_type 'mixtral'"):
        translate_config(dict(LLAMA2_7B, model_type="mixtral"))
    with pytest.raises(Unsupported, match="activation"):
        translate_config(dict(LLAMA2_7B, hidden_act="quick_gelu"))


def test_vocab_byte_level_and_sentencepiece(tmp_path):
    bl = {"model": {"type": "BPE", "vocab": {"Hello": 0, "Ġworld": 1, "Ċ": 2, "ä½": 3}},
          "added_tokens": [{"id": 4, "content": "<|endoftext|>"}], "decoder": {"type": "ByteLevel"}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(bl))
    assert vocab.load(tmp_path, 6) == ["Hello", " world", "\n", "\\xe4\\xbd", "<|endoftext|>", "<|5|>"]
    sp = {"model": {"type": "BPE", "vocab": {"▁the": 0, "<0x41>": 1, "<0x0A>": 2, "x": 3}},
          "decoder": {"type": "Sequence", "decoders": [{"type": "Replace"}, {"type": "ByteFallback"}]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(sp))
    with pytest.warns(UserWarning, match="beyond"):
        assert vocab.load(tmp_path, 3) == [" the", "A", "<0x0A>"]


def _tiny_llama_folder(path: Path, extra=None):
    path.mkdir()
    cfg = dict(LLAMA2_7B, hidden_size=8, intermediate_size=16, num_attention_heads=2, num_key_value_heads=2,
               num_hidden_layers=1, vocab_size=5, max_position_embeddings=16)
    (path / "config.json").write_text(json.dumps(cfg))
    r = lambda *s: torch.randn(*s)
    t = {"model.embed_tokens.weight": r(5, 8), "model.norm.weight": r(8), "lm_head.weight": r(5, 8),
         "model.layers.0.input_layernorm.weight": r(8), "model.layers.0.post_attention_layernorm.weight": r(8),
         "model.layers.0.mlp.gate_proj.weight": r(16, 8), "model.layers.0.mlp.up_proj.weight": r(16, 8),
         "model.layers.0.mlp.down_proj.weight": r(8, 16), "model.layers.0.self_attn.rotary_emb.inv_freq": r(2)}
    for p in "qkvo":
        t[f"model.layers.0.self_attn.{p}_proj.weight"] = r(8, 8)
    t.update(extra or {})
    save_safetensors(t, path / "model.safetensors")
    return path


def test_convert_writes_engine_names_and_dtypes(tmp_path):
    folder = _tiny_llama_folder(tmp_path / "hf")
    s = convert(folder, tmp_path / "out.safetensors", dtype="bfloat16")
    out = tensors.Shards([tmp_path / "out.safetensors"])
    assert "blocks.0.attn.q_proj.weight" in out and "unembed.weight" in out and "ln_final.weight" in out
    assert out.dtype("blocks.0.mlp.gate.weight") == "BF16" and out.dtype("blocks.0.ln1.weight") == "F32"
    assert not any("inv_freq" in n for n in out.names())  # buffers dropped
    assert s["tensors"] == 12 and not s["config"]["bias"] and not s["vocab"]


def test_convert_refuses_unknown_tensors(tmp_path):
    folder = _tiny_llama_folder(tmp_path / "hf", {"model.layers.0.mlp.experts.0.weight": torch.zeros(2, 2)})
    with pytest.raises(Unsupported, match="no IndraNet counterpart"):
        convert(folder, tmp_path / "out.safetensors")


def test_convert_refuses_part_of_a_bias_part(tmp_path):
    folder = _tiny_llama_folder(tmp_path / "hf", {"model.layers.0.self_attn.q_proj.bias": torch.zeros(8)})
    with pytest.raises(Unsupported, match="'qkv' biases are there but not model|IndraNet biases whole parts"):
        convert(folder, tmp_path / "out.safetensors")
    folder = _tiny_llama_folder(tmp_path / "hf2", {f"model.layers.0.self_attn.{x}_proj.bias": torch.zeros(8) for x in "qkv"})
    assert convert(folder, tmp_path / "out.safetensors")["config"]["bias"] == ["qkv"]


def test_convert_refuses_pickles(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(LLAMA2_7B))
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    with pytest.raises(Unsupported, match="pickle"):
        convert(tmp_path, tmp_path / "out.safetensors")


# --- golden tests against transformers ----------------------------------------------------------

V, T = 97, 12
SIZES = dict(vocab_size=V, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_hidden_layers=3,
             max_position_embeddings=64, initializer_range=0.1)


def _hf_config(tf, kind):
    small_window = dict(sliding_window=4)
    if kind == "gpt2":
        return tf.GPT2Config(vocab_size=V, n_embd=64, n_layer=3, n_head=4, n_positions=64, initializer_range=0.1)
    if kind.startswith("gpt_neox"):
        return tf.GPTNeoXConfig(**SIZES, rotary_pct=0.5, use_parallel_residual=kind == "gpt_neox",
                                tie_word_embeddings=False)
    table = {
        "llama": (tf.LlamaConfig, dict(num_key_value_heads=2)),
        "llama_tied_bias": (tf.LlamaConfig, dict(tie_word_embeddings=True, attention_bias=True, mlp_bias=True, head_dim=24)),
        "mistral": (tf.MistralConfig, dict(num_key_value_heads=2, **small_window)),
        "qwen2": (tf.Qwen2Config, dict(num_key_value_heads=2, use_sliding_window=True, max_window_layers=1, **small_window)),
        "qwen3": (getattr(tf, "Qwen3Config", None), dict(num_key_value_heads=2, head_dim=24)),
        "phi3": (tf.Phi3Config, dict(num_key_value_heads=2, pad_token_id=0, partial_rotary_factor=0.5, **small_window)),
        "gemma": (tf.GemmaConfig, dict(num_key_value_heads=1, head_dim=16)),
        "gemma2": (tf.Gemma2Config, dict(num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=20,
                                         attn_logit_softcapping=5.0, final_logit_softcapping=10.0, **small_window)),
        "gemma3": (getattr(tf, "Gemma3TextConfig", None), dict(num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=20,
                                                               sliding_window_pattern=2, rope_local_base_freq=100.0,
                                                               **small_window)),
        # RoPE scaling. original_max_position_embeddings 16 puts head size 16's frequencies on both sides of
        # llama3's and yarn's thresholds within the 12 tokens tested.
        "llama31": (tf.LlamaConfig, dict(num_key_value_heads=2, rope_scaling=dict(
            rope_type="llama3", factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=16))),
        "llama_linear": (tf.LlamaConfig, dict(rope_scaling=dict(rope_type="linear", factor=4.0))),
        "llama_yarn": (tf.LlamaConfig, dict(rope_scaling=dict(rope_type="yarn", factor=4.0, original_max_position_embeddings=16))),
        "llama_dynamic": (tf.LlamaConfig, dict(rope_scaling=dict(rope_type="dynamic", factor=2.0))),
        "phi3_longrope": (tf.Phi3Config, dict(num_key_value_heads=2, pad_token_id=0, partial_rotary_factor=0.5,
                                              original_max_position_embeddings=16, rope_scaling=dict(
                                                  type="longrope", short_factor=[1.0, 1.4, 2.2, 3.5],
                                                  long_factor=[2.0, 3.0, 5.0, 8.0]))),
        "gemma3_linear": (getattr(tf, "Gemma3TextConfig", None), dict(
            num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=20, sliding_window_pattern=2,
            rope_local_base_freq=100.0, rope_scaling=dict(rope_type="linear", factor=8.0), **small_window)),
    }
    cls, kw = table[kind]
    if cls is None:
        pytest.skip(f"this transformers has no {kind}")
    return cls(**SIZES, **kw)


SCALED = ["llama31", "llama_linear", "llama_yarn", "llama_dynamic", "phi3_longrope", "gemma3_linear"]
KINDS = ["llama", "llama_tied_bias", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "gpt2",
         "gpt_neox", "gpt_neox_sequential"] + SCALED


@pytest.mark.parametrize("kind", [k for k in SCALED if k != "gemma3_linear"])
def test_rope_frequencies_match_transformers(kind):
    """indranet's frequencies and cos/sin factor for the converted config = transformers' own."""
    tf = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from indranet.layers.positional import rope_frequencies
    hf = _hf_config(tf, kind)
    kind_name = hf.rope_scaling.get("rope_type", hf.rope_scaling.get("type"))
    theirs, their_factor = ROPE_INIT_FUNCTIONS[kind_name](hf, None)
    c = translate_config(hf.to_dict())
    rot = int(c["d_head"] * c["rope_fraction"])
    ours, factor, _ = rope_frequencies(c["rope_theta"], rot, c["rope_scaling"])
    assert torch.allclose(ours, theirs, rtol=1e-6, atol=0) and factor == pytest.approx(their_factor, rel=1e-12)
    assert not torch.allclose(ours, rope_frequencies(c["rope_theta"], rot, None)[0]) or kind == "llama_dynamic"


def test_engine_refuses_sequences_past_the_scaling_limit(tools, tmp_path):
    """longrope (and dynamic) change their frequencies past original_max_position_embeddings: refused."""
    _hf_model("phi3_longrope", tmp_path / "hf", torch.float32)
    convert(tmp_path / "hf", tmp_path / "m.safetensors")
    run = lambda n: subprocess.run([str(tools / "indra"), "run", str(tmp_path / "m.safetensors"), "--tokens",
                                    ",".join(["5"] * n)], capture_output=True, text=True)
    assert run(16).returncode == 0
    r = run(17)
    assert r.returncode != 0 and "original_max_position_embeddings=16" in r.stderr, r.stdout + r.stderr


@pytest.fixture(scope="module")
def tools():
    if shutil.which("make") is None or shutil.which("c++") is None:
        pytest.skip("needs make and a C++ compiler")
    subprocess.run(["make", "-s", "-C", str(ROOT), "all"], check=True)
    return ROOT / "build"


def _hf_model(kind, folder: Path, dtype):
    tf = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = tf.AutoModelForCausalLM.from_config(_hf_config(tf, kind), attn_implementation="eager").eval()
    with torch.no_grad():
        for p in model.parameters():  # norm weights and biases off their 0/1 init, so mapping mistakes show
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.2)
        if dtype == torch.bfloat16:  # the reference: bf16-rounded weights, computed in float32
            for p in model.parameters():
                p.copy_(p.bfloat16().float())
    # Saved from a copy: .to(bfloat16) would also round buffers such as RoPE's cached frequencies.
    copy.deepcopy(model).to(dtype).save_pretrained(folder, safe_serialization=True)
    return model


def _reference(model, n_layers, path: Path):
    tokens = torch.randint(0, V, (T,))
    with torch.no_grad():
        out = model(tokens[None], output_hidden_states=True)
    t = {"tokens": tokens.long(), "logits": out.logits[0].float()}
    for i in range(n_layers):  # hidden_states[i] is block i's input; the last one is after the final norm
        t[f"cache.blocks.{i}.hook_resid_pre"] = out.hidden_states[i][0].float()
    save_safetensors(t, path, {"format": "indranet-reference", "partial": "1"})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("kind", KINDS)
def test_engine_matches_transformers(tools, tmp_path, kind, dtype):
    model = _hf_model(kind, tmp_path / "hf", dtype)
    s = convert(tmp_path / "hf", tmp_path / "model.safetensors")
    expected_bias = {"qwen2": ["qkv"], "llama_tied_bias": True, "gpt2": True, "gpt_neox": True, "gpt_neox_sequential": True}
    assert s["config"]["bias"] == expected_bias.get(kind, False)
    _reference(model, s["config"]["n_layers"], tmp_path / "ref.safetensors")
    for fmt in ["fp32", "fp16"] if dtype == torch.bfloat16 else ["fp32"]:
        r = subprocess.run([str(tools / "tfi_golden"), str(tmp_path / "model.safetensors"), str(tmp_path / "ref.safetensors"),
                            "1e-4", fmt], capture_output=True, text=True)
        assert r.returncode == 0, f"{kind} {fmt}\n{r.stdout}{r.stderr}"


def test_cli(tools, tmp_path):
    _hf_model("qwen2", tmp_path / "hf", torch.float32)
    run = lambda *a: subprocess.run([sys.executable, "-m", "indranet_hf", *a], capture_output=True, text=True,
                                    cwd=ROOT, env={"PYTHONPATH": str(ROOT / "python"), "PATH": ""})
    r = run(str(tmp_path / "hf"), "--config-only")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["sliding_window"] == 4
    r = run(str(tmp_path / "hf"), str(tmp_path / "m.safetensors"), "--dtype", "float16")
    assert r.returncode == 0 and "qwen2 -> " in r.stdout, r.stderr
    info = subprocess.run([str(tools / "indra"), "info", str(tmp_path / "m.safetensors")], capture_output=True, text=True)
    assert info.returncode == 0 and "layers 3" in info.stdout, info.stdout + info.stderr
    r = run(str(tmp_path / "nowhere"), str(tmp_path / "x.safetensors"))
    assert r.returncode == 1 and "error:" in r.stderr



# --- the engine's tokenizer against Hugging Face's tokenizers library -------------------------------

TOKENIZER_TEXTS = [
    "Hello world", " Hello  world ", " ", "I'm sure it's fine, we'll see. THEY'RE HERE'S", "don't 'tis o'clock",
    "12345678901 and 3.14159 and 1,000,000", "a\tb\t\tc", "   leading and trailing   ", "naïve café résumé",
    "naïve café (decomposed)", "日本語のテキストと中文文本", "한국어 텍스트 각", "مرحبا بالعالم", "नमस्ते दुनिया",
    "emoji 😀👍🏽 family 👨‍👩‍👧 flags 🇫🇷", "non breaking　spaces em", "def f(x): return x**2  # square",
    "https://example.com/path?q=1&r=2", "!!!??...---***", "<|endoftext|> after special", "before<|endoftext|>after",
    "<s>Hello</s>", "x" * 200, "ab" * 60, " " * 40 + "x", "Ünïcödé ÀÉÎÕÜ ǅ ſ ﬁ", "Zero​width‍join",
    "MiXeD CaSe 'S 'T 'Re", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛 ①②③ ½ Ⅻ", "́ leading combining mark", "a <tool>call</tool> [MASK]  here",
    "say hello, not hellooo or ohello", "the quick brown fox jumps over the lazy dog",
]
MULTILINE_TEXTS = ["line one\nline two\n\n\nline five\r\nend", "def f(x):\n    return x\n", "\n", "a\n b \n\n c"]
CORPUS = ["the quick brown fox jumps over the lazy dog", "I'm sure it's fine, we'll see", "hello world hello there",
          "numbers 123 456 7890 and 3.14", "naïve café résumé über straße", "日本語 中文 한국어", "def f(x): return x**2",
          "  indented   text  with   spaces  ", "emoji 😀 and symbols !?.,;:"] * 3

LLAMA3_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN_SPLIT = LLAMA3_SPLIT.replace(r"\p{N}{1,3}", r"\p{N}")


def _tokenizer(style):
    """A small tokenizer of the given style, trained by the tokenizers library on CORPUS."""
    tk = pytest.importorskip("tokenizers")
    from tokenizers import AddedToken, Regex, decoders, models, normalizers, pre_tokenizers, processors, trainers
    byte_level = style not in ("llama2_legacy", "llama2_metaspace", "metaspace_split", "gemma")
    model = models.BPE(byte_fallback=not byte_level, unk_token=None if byte_level else "<unk>", fuse_unk=not byte_level,
                       ignore_merges=style == "llama3")
    t = tk.Tokenizer(model)
    specials = ["<|endoftext|>"] if byte_level else ["<unk>", "<s>", "</s>"]
    if style in ("gpt2", "gpt2_prefix", "added", "roberta", "bert"):
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=style == "gpt2_prefix")
    elif style in ("llama3", "qwen"):
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(LLAMA3_SPLIT if style == "llama3" else QWEN_SPLIT), "isolated"),
                                                   pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        if style == "qwen":
            t.normalizer = normalizers.NFC()
    elif style == "smollm":
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
    elif style.startswith("split_"):
        behavior = style[len("split_"):].replace("_invert", "")
        t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(" ", behavior, invert=style.endswith("_invert")),
                                                   pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    elif style == "llama2_legacy":
        t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    elif style == "llama2_metaspace":
        t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=False)
    elif style == "metaspace_split":
        t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    elif style == "gemma":
        t.normalizer = normalizers.Replace(" ", "▁")
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=specials, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if byte_level else [])
    t.train_from_iterator(CORPUS, trainer)
    if style == "added":
        t.add_tokens([AddedToken("<tool>", lstrip=True), AddedToken("</tool>"), AddedToken("[MASK]", rstrip=True),
                      AddedToken("hello", single_word=True, normalized=True)])
    if style in ("llama3", "llama2_legacy", "llama2_metaspace", "gemma"):
        bos = "<|endoftext|>" if byte_level else "<s>"
        t.post_processor = processors.TemplateProcessing(single=f"{bos} $A", special_tokens=[(bos, t.token_to_id(bos))])
    elif style == "roberta":
        eot = t.token_to_id("<|endoftext|>")
        t.post_processor = processors.RobertaProcessing(("<|endoftext|>", eot), ("<|endoftext|>", eot))
    elif style == "bert":
        eot = t.token_to_id("<|endoftext|>")
        t.post_processor = processors.BertProcessing(("<|endoftext|>", eot), ("<|endoftext|>", eot))
    else:
        t.post_processor = processors.ByteLevel(trim_offsets=False) if byte_level else None
    d = json.loads(t.to_str())
    if not byte_level:  # SentencePiece-style vocabularies hold all 256 byte tokens, for byte fallback
        vocab = d["model"]["vocab"]
        for b in range(256):
            vocab.setdefault(f"<0x{b:02X}>", len(vocab))
    return tk.Tokenizer.from_str(json.dumps(d)), d


TOKENIZER_STYLES = ["gpt2", "gpt2_prefix", "llama3", "qwen", "smollm", "llama2_legacy", "llama2_metaspace", "metaspace_split",
                    "gemma", "added", "roberta", "bert", "split_removed", "split_merged_with_previous",
                    "split_merged_with_next", "split_contiguous", "split_removed_invert"]


@pytest.mark.parametrize("style", TOKENIZER_STYLES)
def test_engine_tokenizer_matches_tokenizers(tools, tmp_path, style):
    """indra tokenize = tokenizers' encode().ids, with and without special tokens, for every kind of
    tokenizer the engine implements, stored the way convert() stores it."""
    from indranet_hf import vocab as vocab_mod
    hf, d = _tokenizer(style)
    folder = tmp_path / "hf"
    folder.mkdir()
    (folder / "tokenizer.json").write_text(json.dumps(d), encoding="utf-8")
    stored = vocab_mod.tokenizer(folder)
    assert stored is not None
    save_safetensors({}, tmp_path / "m.safetensors", {"tokenizer": json.dumps(stored, separators=(",", ":"))})
    (tmp_path / "texts.txt").write_text("\n".join(TOKENIZER_TEXTS) + "\n", encoding="utf-8")
    for special in (True, False):
        args = [str(tools / "indra"), "tokenize", str(tmp_path / "m.safetensors"), "--text-file", str(tmp_path / "texts.txt")]
        r = subprocess.run(args + ([] if special else ["--no-special-tokens"]), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        got = r.stdout.split("\n")
        for i, text in enumerate(TOKENIZER_TEXTS):
            assert got[i] == ",".join(map(str, hf.encode(text, add_special_tokens=special).ids)), f"{style} {special} {text!r}"
    for text in MULTILINE_TEXTS:  # --text keeps newlines
        r = subprocess.run([str(tools / "indra"), "tokenize", str(tmp_path / "m.safetensors"), "--text", text],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        last = r.stdout.rstrip("\n").split("\n")[-1]
        assert last.rstrip() == ("--tokens " + ",".join(map(str, hf.encode(text).ids))).rstrip(), f"{style} {text!r}"


def test_unsupported_tokenizer_is_not_stored(tmp_path):
    d = {"model": {"type": "Unigram", "vocab": [["a", 0.0]]}, "normalizer": None, "pre_tokenizer": None}
    (tmp_path / "tokenizer.json").write_text(json.dumps(d))
    from indranet_hf import vocab as vocab_mod
    with pytest.warns(UserWarning, match="Unigram"):
        assert vocab_mod.tokenizer(tmp_path) is None
    d = {"model": {"type": "BPE", "vocab": {}, "merges": []}, "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(d))
    with pytest.warns(UserWarning, match="NFKC"):
        assert vocab_mod.tokenizer(tmp_path) is None
