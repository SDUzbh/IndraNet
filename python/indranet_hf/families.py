"""Each supported Hugging Face model family: how its config.json maps onto IndraNet's config, and
which of its tensors become which IndraNet tensors.

Some of a family's architecture is not in its config.json but in its modeling code (Gemma's
``1 + w`` RMSNorm, Qwen2's q/k/v-only biases, GPT-2's transposed Conv1D weights, GPT-NeoX's
per-head fused q/k/v). Those rules live here, one place per family.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Unsupported(ValueError):
    """The checkpoint uses something IndraNet's engine does not implement."""


@dataclass
class Out:
    """One IndraNet tensor: taken from HF tensor ``src``, rearranged by ``pick`` (rows, transposes:
    works on any dtype, bf16 bits included), then, for a 1-D tensor, ``add`` added to its values."""
    src: str
    pick: Optional[Callable[[np.ndarray], np.ndarray]] = None
    add: float = 0.0


# Every field of IndraNet's ModelConfig that the engine reads (the training-only ones are left out).
CONFIG_FIELDS = (
    "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_head", "d_mlp", "mlp_multiple_of",
    "max_seq_len", "norm", "norm_eps", "norm_position", "positional", "rope_theta", "rope_fraction", "rope_scaling",
    "attention", "mlp", "activation", "qk_norm", "sliding_window", "attn_scale", "attn_logit_softcap",
    "parallel_residual", "bias", "tie_embeddings", "embed_scale", "final_logit_softcap", "layer_overrides",
)
PER_LAYER = ("sliding_window", "rope_theta", "rope_scaling")  # what can differ between layers of a converted model

ACTIVATIONS = {
    "silu": "silu", "swish": "silu", "gelu": "gelu", "gelu_new": "gelu_tanh", "gelu_pytorch_tanh": "gelu_tanh",
    "gelu_fast": "gelu_tanh", "relu": "relu", "tanh": "tanh", "relu2": "relu2",
}

# HF tensors that are buffers, not weights (attention masks, cached rotary frequencies).
IGNORED = re.compile(r"(rotary_emb\.inv_freq|\.attn\.bias|\.attn\.masked_bias|attention\.bias|attention\.masked_bias)$")


def activation(name: Optional[str]) -> str:
    if name not in ACTIVATIONS:
        raise Unsupported(f"activation {name!r} has no IndraNet implementation; supported: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]


def base_config(**kw) -> dict:
    cfg = dict(vocab_size=0, d_model=0, n_layers=0, n_heads=0, n_kv_heads=None, d_head=None, d_mlp=None,
               mlp_multiple_of=64, max_seq_len=2048, norm="rmsnorm", norm_eps=1e-6, norm_position="pre",
               positional="rope", rope_theta=10000.0, rope_fraction=1.0, rope_scaling=None, attention="causal",
               mlp="swiglu",
               activation="silu", qk_norm=False, sliding_window=None, attn_scale=None, attn_logit_softcap=None,
               parallel_residual=False, bias=False, tie_embeddings=False, embed_scale=False,
               final_logit_softcap=None, layer_overrides={})
    cfg.update(kw)
    assert tuple(cfg) == CONFIG_FIELDS
    return cfg


def rope(hf: dict, layer_type: str = "full_attention") -> Tuple[float, float, Optional[dict]]:
    """(theta, rotated fraction, IndraNet rope_scaling) for one layer type. Reads both the older config
    keys (rope_theta, rope_scaling, rotary_pct, rope_local_base_freq) and the newer rope_parameters."""
    rp = hf.get("rope_parameters")
    local = None
    if isinstance(rp, dict) and isinstance(rp.get(layer_type), dict):
        rp = rp[layer_type]  # one set per layer type (Gemma 3)
    elif rp is None:
        rp = hf.get("rope_scaling") or {}
        if layer_type == "sliding_attention" and "rope_local_base_freq" in hf:
            local, rp = hf["rope_local_base_freq"], {}  # Gemma 3: the scaling is for global layers only
    theta = local or rp.get("rope_theta") or hf.get("rope_theta") or hf.get("rotary_emb_base") or 10000.0
    frac = rp.get("partial_rotary_factor", hf.get("partial_rotary_factor", hf.get("rotary_pct", 1.0)))
    return float(theta), float(frac), rope_scaling(hf, rp)


def rope_scaling(hf: dict, rp: dict) -> Optional[dict]:
    """IndraNet's rope_scaling for a Hugging Face one, with the values transformers 4.57 reads (its
    modeling_rope_utils): the defaults it falls back on are made explicit."""
    kind = rp.get("rope_type", rp.get("type", "default"))
    kind = {"su": "longrope"}.get(kind, kind)  # Phi-3's older name
    keep = lambda *keys: {k: rp[k] for k in keys if rp.get(k) is not None}
    max_pos = hf.get("max_position_embeddings")
    if kind == "default" or (kind == "linear" and float(rp.get("factor", 1.0)) == 1.0):
        return None
    if kind == "linear":
        return {"type": "linear", "factor": float(rp["factor"])}
    if kind == "dynamic":  # transformers: plain RoPE until a sequence passes max_position_embeddings
        return {"type": "dynamic", "factor": float(rp["factor"]), "original_max_position_embeddings": max_pos}
    if kind == "llama3":
        return {"type": "llama3", **keep("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")}
    if kind == "yarn":
        return {"type": "yarn", "factor": float(rp["factor"]),
                "original_max_position_embeddings": rp.get("original_max_position_embeddings") or max_pos,
                **keep("beta_fast", "beta_slow", "mscale", "mscale_all_dim", "attention_factor", "truncate")}
    if kind == "longrope":
        out = {"type": "longrope", "short_factor": [float(f) for f in rp["short_factor"]]}
        orig = hf.get("original_max_position_embeddings")
        if orig:  # transformers then takes the factor from the two lengths
            out.update(original_max_position_embeddings=orig, factor=max_pos / orig)
        else:
            out.update(original_max_position_embeddings=max_pos, **keep("factor"))
        out.update(keep("attention_factor"))
        return out
    raise Unsupported(f"RoPE scaling {kind!r} (rope_scaling / rope_parameters) is not implemented by IndraNet's engine; "
                      "supported: linear, dynamic, llama3, yarn, longrope")


def with_layers(cfg: dict, layers: Sequence[dict]) -> dict:
    """Set the per-layer fields: the most common setting model-wide, the rest as layer_overrides."""
    keys = [json.dumps(l, sort_keys=True) for l in layers]  # dicts (rope_scaling) are not hashable
    common = json.loads(Counter(keys).most_common(1)[0][0])
    cfg.update(common)
    cfg["layer_overrides"] = {}
    for i, l in enumerate(layers):
        diff = {name: v for name, v in l.items() if common[name] != v}
        if diff:
            cfg["layer_overrides"][str(i)] = diff
    return cfg


def rows(lo: int, hi: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: a[lo:hi]


def find_prefix(names: Sequence[str], suffix: str) -> str:
    hits = [n[:-len(suffix)] for n in names if n.endswith(suffix) and (n == suffix or n[-len(suffix) - 1] == ".")]
    if len(hits) != 1:
        raise Unsupported(f"expected one tensor named '*{suffix}', found {len(hits)}")
    return hits[0]


def optional(outs: Dict[str, Out], names, ours: str, theirs: str, **kw) -> None:
    if theirs in names:
        outs[ours] = Out(theirs, **kw)


def unembed(outs: Dict[str, Out], cfg: dict, names, candidates: Sequence[str]) -> None:
    """The output matrix: its own tensor unless tied (a tied copy in the file is not needed)."""
    if cfg["tie_embeddings"]:
        return
    for c in candidates:
        if c in names:
            outs["unembed.weight"] = Out(c)
            return
    raise Unsupported(f"the config says the output matrix is not tied, but there is no {candidates[0]!r}")


# --- Llama and the families built like it --------------------------------------------------------

@dataclass
class LlamaStyle:
    """Llama, Mistral, Qwen2/3, Phi-3 and Gemma 1/2/3: pre-norm RMSNorm decoders with RoPE and gated MLPs."""
    windows: str = "none"          # which layers use config.sliding_window: none | all | qwen | gemma2 | gemma3
    qk_norm: bool = False          # per-head q/k RMSNorm (Qwen3, Gemma 3)
    fused: bool = False            # qkv_proj and gate_up_proj (Phi-3)
    gemma: int = 0                 # Gemma version: 1+w norms, scaled embeddings, GeGLU; 2+: sandwich norms
    tie: bool = False              # tie_word_embeddings when the config doesn't say

    def layer_types(self, hf: dict) -> List[str]:
        n = hf["num_hidden_layers"]
        if hf.get("layer_types"):
            return list(hf["layer_types"])
        if self.windows == "all":
            sliding = [True] * n
        elif self.windows == "qwen":
            sliding = [i >= hf.get("max_window_layers", 0) for i in range(n)]
        elif self.windows == "gemma2":
            sliding = [(i + 1) % 2 != 0 for i in range(n)]
        elif self.windows == "gemma3":
            pattern = hf.get("sliding_window_pattern", 6)
            sliding = [(i + 1) % pattern != 0 for i in range(n)]
        else:
            sliding = [False] * n
        return ["sliding_attention" if s else "full_attention" for s in sliding]

    def window(self, hf: dict) -> Optional[int]:
        if self.windows == "none" or (self.windows == "qwen" and not hf.get("use_sliding_window")):
            return None
        return hf.get("sliding_window")

    def config(self, hf: dict) -> dict:
        d, H = hf["hidden_size"], hf["num_attention_heads"]
        D = hf.get("head_dim") or d // H
        act = activation((hf.get("hidden_activation") or "gelu_pytorch_tanh") if self.gemma else hf.get("hidden_act", "silu"))
        mlp = "swiglu" if act == "silu" else "geglu" if act.startswith("gelu") else "glu"
        scale = None
        if self.gemma >= 2 and hf.get("query_pre_attn_scalar") not in (None, D):
            scale = float(hf["query_pre_attn_scalar"]) ** -0.5
        cfg = base_config(
            vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"], n_heads=H,
            n_kv_heads=hf.get("num_key_value_heads") or H, d_head=D, d_mlp=hf["intermediate_size"],
            max_seq_len=hf.get("max_position_embeddings") or 2048, norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            norm_position="sandwich" if self.gemma >= 2 else "pre", mlp=mlp, activation=act, qk_norm=self.qk_norm,
            attn_scale=scale, attn_logit_softcap=hf.get("attn_logit_softcapping") if self.gemma >= 2 else None,
            final_logit_softcap=hf.get("final_logit_softcapping") if self.gemma >= 2 else None,
            tie_embeddings=bool(hf.get("tie_word_embeddings", self.tie)), embed_scale=self.gemma > 0)
        window, layers = self.window(hf), []
        for t in self.layer_types(hf):
            theta, frac, scaling = rope(hf, t)
            layers.append({"sliding_window": window if t == "sliding_attention" else None, "rope_theta": theta,
                           "rope_scaling": scaling})
            cfg["rope_fraction"] = frac
        return with_layers(cfg, layers)

    def tensors(self, cfg: dict, names) -> Dict[str, Out]:
        p = find_prefix(names, "embed_tokens.weight")
        one = 1.0 if self.gemma else 0.0
        outs: Dict[str, Out] = {"embed.weight": Out(p + "embed_tokens.weight")}
        for i in range(cfg["n_layers"]):
            c = {**cfg, **cfg["layer_overrides"].get(str(i), {})}
            L, B = f"{p}layers.{i}.", f"blocks.{i}."
            outs[B + "ln1.weight"] = Out(L + "input_layernorm.weight", add=one)
            if self.gemma >= 2:
                outs[B + "ln1_post.weight"] = Out(L + "post_attention_layernorm.weight", add=one)
            q, kv = c["n_heads"] * c["d_head"], c["n_kv_heads"] * c["d_head"]
            a = L + "self_attn."
            if self.fused:
                for part, (lo, hi) in {"q": (0, q), "k": (q, q + kv), "v": (q + kv, q + 2 * kv)}.items():
                    outs[f"{B}attn.{part}_proj.weight"] = Out(a + "qkv_proj.weight", rows(lo, hi))
                    optional(outs, names, f"{B}attn.{part}_proj.bias", a + "qkv_proj.bias", pick=rows(lo, hi))
            for part in ("q", "k", "v", "o") if not self.fused else ("o",):
                outs[f"{B}attn.{part}_proj.weight"] = Out(f"{a}{part}_proj.weight")
                optional(outs, names, f"{B}attn.{part}_proj.bias", f"{a}{part}_proj.bias")
            if self.qk_norm:
                outs[B + "attn.q_norm.weight"] = Out(a + "q_norm.weight", add=one)
                outs[B + "attn.k_norm.weight"] = Out(a + "k_norm.weight", add=one)
            m, F = L + "mlp.", c["d_mlp"]
            if self.fused:
                outs[B + "mlp.gate.weight"] = Out(m + "gate_up_proj.weight", rows(0, F))
                outs[B + "mlp.up.weight"] = Out(m + "gate_up_proj.weight", rows(F, 2 * F))
            for ours, theirs in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                if not self.fused or ours == "down":
                    outs[f"{B}mlp.{ours}.weight"] = Out(f"{m}{theirs}.weight")
                    optional(outs, names, f"{B}mlp.{ours}.bias", f"{m}{theirs}.bias")
            if self.gemma >= 2:
                outs[B + "ln2.weight"] = Out(L + "pre_feedforward_layernorm.weight", add=one)
                outs[B + "ln2_post.weight"] = Out(L + "post_feedforward_layernorm.weight", add=one)
            else:
                outs[B + "ln2.weight"] = Out(L + "post_attention_layernorm.weight", add=one)
        outs["ln_final.weight"] = Out(p + "norm.weight", add=one)
        top = p[:-len("model.")] if p.endswith("model.") else p
        unembed(outs, cfg, names, [top + "lm_head.weight", "lm_head.weight"])
        return outs


# --- GPT-2 ---------------------------------------------------------------------------------------

class GPT2:
    def config(self, hf: dict) -> dict:
        if hf.get("scale_attn_by_inverse_layer_idx") or hf.get("reorder_and_upcast_attn"):
            raise Unsupported("GPT-2's scale_attn_by_inverse_layer_idx / reorder_and_upcast_attn are not implemented")
        d = hf["n_embd"]
        return base_config(
            vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["n_layer"], n_heads=hf["n_head"], n_kv_heads=hf["n_head"],
            d_head=d // hf["n_head"], d_mlp=hf.get("n_inner") or 4 * d, max_seq_len=hf["n_positions"], norm="layernorm",
            norm_eps=float(hf.get("layer_norm_epsilon", 1e-5)), positional="learned", mlp="mlp",
            activation=activation(hf.get("activation_function", "gelu_new")),
            attn_scale=None if hf.get("scale_attn_weights", True) else 1.0,
            tie_embeddings=bool(hf.get("tie_word_embeddings", True)))

    def tensors(self, cfg: dict, names) -> Dict[str, Out]:
        p, d = find_prefix(names, "wte.weight"), cfg["d_model"]
        T = lambda a: a.T  # Conv1D stores (in, out)
        outs = {"embed.weight": Out(p + "wte.weight"), "pos_embed.weight": Out(p + "wpe.weight")}
        for i in range(cfg["n_layers"]):
            L, B = f"{p}h.{i}.", f"blocks.{i}."
            for n in ("weight", "bias"):
                outs[f"{B}ln1.{n}"] = Out(f"{L}ln_1.{n}")
            for j, part in enumerate("qkv"):
                outs[f"{B}attn.{part}_proj.weight"] = Out(L + "attn.c_attn.weight", lambda a, j=j: a[:, j * d:(j + 1) * d].T)
                outs[f"{B}attn.{part}_proj.bias"] = Out(L + "attn.c_attn.bias", rows(j * d, (j + 1) * d))
            outs[B + "attn.o_proj.weight"] = Out(L + "attn.c_proj.weight", T)
            outs[B + "attn.o_proj.bias"] = Out(L + "attn.c_proj.bias")
            outs[B + "mlp.up.weight"] = Out(L + "mlp.c_fc.weight", T)
            outs[B + "mlp.up.bias"] = Out(L + "mlp.c_fc.bias")
            outs[B + "mlp.down.weight"] = Out(L + "mlp.c_proj.weight", T)
            outs[B + "mlp.down.bias"] = Out(L + "mlp.c_proj.bias")
            for n in ("weight", "bias"):
                outs[f"{B}ln2.{n}"] = Out(f"{L}ln_2.{n}")
        for n in ("weight", "bias"):
            outs[f"ln_final.{n}"] = Out(f"{p}ln_f.{n}")
        unembed(outs, cfg, names, ["lm_head.weight"])
        return outs


# --- GPT-NeoX (Pythia) -----------------------------------------------------------------------------

class NeoX:
    def config(self, hf: dict) -> dict:
        theta, frac, scaling = rope(hf)
        d, H = hf["hidden_size"], hf["num_attention_heads"]
        return base_config(
            vocab_size=hf["vocab_size"], d_model=d, n_layers=hf["num_hidden_layers"], n_heads=H, n_kv_heads=H,
            d_head=d // H, d_mlp=hf["intermediate_size"], max_seq_len=hf.get("max_position_embeddings") or 2048,
            norm="layernorm", norm_eps=float(hf.get("layer_norm_eps", 1e-5)), rope_theta=theta, rope_fraction=frac,
            rope_scaling=scaling,
            mlp="mlp", activation=activation(hf.get("hidden_act", "gelu")),
            parallel_residual=bool(hf.get("use_parallel_residual", True)),
            tie_embeddings=bool(hf.get("tie_word_embeddings", False)))

    def tensors(self, cfg: dict, names) -> Dict[str, Out]:
        p = find_prefix(names, "embed_in.weight")
        H, D, d = cfg["n_heads"], cfg["d_head"], cfg["d_model"]
        outs = {"embed.weight": Out(p + "embed_in.weight")}
        for i in range(cfg["n_layers"]):
            L, B = f"{p}layers.{i}.", f"blocks.{i}."
            for n in ("weight", "bias"):
                outs[f"{B}ln1.{n}"] = Out(f"{L}input_layernorm.{n}")
                outs[f"{B}ln2.{n}"] = Out(f"{L}post_attention_layernorm.{n}")
            # query_key_value's rows are per head: head h's q, then its k, then its v.
            for j, part in enumerate("qkv"):
                outs[f"{B}attn.{part}_proj.weight"] = Out(L + "attention.query_key_value.weight",
                                                         lambda a, j=j: a.reshape(H, 3, D, d)[:, j].reshape(H * D, d))
                optional(outs, names, f"{B}attn.{part}_proj.bias", L + "attention.query_key_value.bias",
                         pick=lambda a, j=j: a.reshape(H, 3, D)[:, j].reshape(H * D))
            for ours, theirs in (("attn.o_proj", "attention.dense"), ("mlp.up", "mlp.dense_h_to_4h"),
                                 ("mlp.down", "mlp.dense_4h_to_h")):
                outs[f"{B}{ours}.weight"] = Out(f"{L}{theirs}.weight")
                optional(outs, names, f"{B}{ours}.bias", f"{L}{theirs}.bias")
        for n in ("weight", "bias"):
            outs[f"ln_final.{n}"] = Out(f"{p}final_layer_norm.{n}")
        unembed(outs, cfg, names, ["embed_out.weight"])
        return outs


FAMILIES = {
    "llama": LlamaStyle(),
    "mistral": LlamaStyle(windows="all"),
    "qwen2": LlamaStyle(windows="qwen"),
    "qwen3": LlamaStyle(windows="qwen", qk_norm=True),
    "phi3": LlamaStyle(windows="all", fused=True),
    "gemma": LlamaStyle(gemma=1, tie=True),
    "gemma2": LlamaStyle(windows="gemma2", gemma=2, tie=True),
    "gemma3_text": LlamaStyle(windows="gemma3", gemma=3, qk_norm=True, tie=True),
    "gpt2": GPT2(),
    "gpt_neox": NeoX(),
}


def family(hf: dict):
    t = hf.get("model_type")
    if t not in FAMILIES:
        raise Unsupported(f"model_type {t!r} is not supported; supported: {', '.join(sorted(FAMILIES))}")
    return FAMILIES[t]


BIAS_PARTS = ("qkv", "o", "mlp", "norm")  # as indranet.config.BIAS_PARTS


def bias_setting(cfg: dict, outs: Dict[str, Out]):
    """IndraNet's ``bias`` for the bias tensors the checkpoint has: True, False, or the parts that
    have them (Qwen2: ["qkv"]). A part counts when every layer has all its biases; a part only partly
    there is refused. A part with nothing to bias (RMSNorm's "norm", no MLP) doesn't count against True."""
    def part_names(i: int) -> Dict[str, List[str]]:
        c, B = {**cfg, **cfg["layer_overrides"].get(str(i), {})}, f"blocks.{i}."
        mlp = [] if c["mlp"] == "none" else ["up", "down"] + (["gate"] if c["mlp"] != "mlp" else [])
        norms = []
        if c["norm"] == "layernorm":
            norms = ["ln1"] + (["ln2"] if mlp else []) + (["ln1_post"] + (["ln2_post"] if mlp else [])
                                                         if c["norm_position"] == "sandwich" else [])
            norms += ["attn.q_norm", "attn.k_norm"] if c["qk_norm"] else []
        return {"qkv": [f"{B}attn.{x}_proj.bias" for x in "qkv"], "o": [f"{B}attn.o_proj.bias"],
                "mlp": [f"{B}mlp.{x}.bias" for x in mlp], "norm": [f"{B}{x}.bias" for x in norms]}
    found = {p: set() for p in BIAS_PARTS}  # per part: the answers seen, True / False
    for i in range(cfg["n_layers"]):
        for part, names in part_names(i).items():
            if names:
                have = [n in outs for n in names]
                if any(have) and not all(have):
                    missing = [n for n, h in zip(names, have) if not h]
                    raise Unsupported(f"some '{part}' biases are there but not {missing[0]}: IndraNet biases whole parts")
                found[part].add(all(have))
    if cfg["norm"] == "layernorm":
        found["norm"].add("ln_final.bias" in outs)
    mixed = [p for p, v in found.items() if len(v) > 1]
    if mixed:
        raise Unsupported(f"'{mixed[0]}' biases are in some layers and not others")
    on = [p for p in BIAS_PARTS if found[p] == {True}]
    off = [p for p in BIAS_PARTS if found[p] == {False}]
    return False if not on else True if not off else on
