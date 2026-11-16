"""Positional encodings.

Two kinds:
- ``EmbeddingPositional`` adds a vector to the token embedding once, at the input
  (learned, sinusoidal).
- ``AttentionPositional`` acts inside every attention layer, by rotating q/k
  (RoPE) or adding a bias to the attention scores (ALiBi). ``"none"`` (NoPE) is
  an AttentionPositional that does nothing, so it can be mixed per layer with
  RoPE or ALiBi through ``layer_overrides``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..registry import POSITIONALS


class EmbeddingPositional(nn.Module):
    kind = "embedding"

    def forward(self, positions: Tensor) -> Tensor:  # (T,) -> (T, d_model)
        raise NotImplementedError


class AttentionPositional(nn.Module):
    kind = "attention"

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Transform q or k of shape (B, H, T, D) at the given absolute positions."""
        return x

    def bias(self, q_pos: Tensor, k_pos: Tensor) -> Optional[Tensor]:
        """Additive attention-score bias of shape (H, Tq, Tk), or None."""
        return None


@POSITIONALS.register("learned")
class LearnedPositional(EmbeddingPositional):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.weight = nn.Parameter(torch.empty(cfg.max_seq_len, cfg.d_model))
        nn.init.normal_(self.weight, std=cfg.init_std)

    def forward(self, positions: Tensor) -> Tensor:
        return self.weight[positions]


@POSITIONALS.register("sinusoidal")
class SinusoidalPositional(EmbeddingPositional):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

    def forward(self, positions: Tensor) -> Tensor:
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / half)
        angles = positions.float()[:, None] * freqs[None, :]
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.d_model % 2:
            pe = torch.nn.functional.pad(pe, (0, 1))
        return pe


@POSITIONALS.register("none")
class NoPositional(AttentionPositional):
    def __init__(self, cfg=None):
        super().__init__()


@POSITIONALS.register("rope")
class RotaryPositional(AttentionPositional):
    """Rotary embeddings, "rotate half" convention (GPT-NeoX / Llama).

    With ``rope_fraction < 1`` only the first part of each head's dims rotate.
    ``rope_scaling`` changes the frequencies and may scale cos and sin (see
    ``rope_frequencies``). Frequencies are recomputed in FP32 on each call rather than
    stored as a buffer, so ``model.bfloat16()`` cannot round them (the error would grow
    with position).
    """

    def __init__(self, cfg):
        super().__init__()
        rot = int(cfg.head_dim * cfg.rope_fraction)
        self.rot_dim = rot - rot % 2
        if self.rot_dim == 0:
            raise ValueError("rope_fraction * head_dim leaves no dimensions to rotate")
        self.theta = cfg.rope_theta
        self.scaling = cfg.rope_scaling

    def inv_freq(self, device: torch.device) -> Tensor:
        return rope_frequencies(self.theta, self.rot_dim, self.scaling, device)[0]

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        inv_freq, factor, max_len = rope_frequencies(self.theta, self.rot_dim, self.scaling, x.device)
        if max_len and positions.numel() and int(positions.max()) >= max_len:
            raise ValueError(f"RoPE scaling '{self.scaling['type']}' is implemented for sequences of up to "
                             f"original_max_position_embeddings={max_len} tokens; longer ones change its frequencies")
        angles = positions.float()[:, None] * inv_freq[None, :]   # (T, rot/2)
        angles = torch.cat([angles, angles], dim=-1)                   # (T, rot)
        cos, sin = angles.cos(), angles.sin()
        if factor != 1.0:
            cos, sin = cos * factor, sin * factor
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        x_rot, x_pass = x[..., : self.rot_dim], x[..., self.rot_dim:]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = x_rot * cos + torch.cat([-x2, x1], dim=-1) * sin
        return torch.cat([rotated, x_pass], dim=-1)


def rope_frequencies(theta: float, rot: int, scaling: Optional[Dict[str, Any]],
                     device: Optional[torch.device] = None) -> Tuple[Tensor, float, int]:
    """(inverse frequencies (rot/2,), cos/sin factor, longest sequence it holds for or 0) for RoPE
    over ``rot`` dims, with ``rope_scaling`` as in ModelConfig. The formulas are Hugging Face
    transformers' (modeling_rope_utils), computed the same way, in float32.

    ``dynamic`` and ``longrope`` change their frequencies once a sequence is longer than
    ``original_max_position_embeddings``; up to that length (the third value) ``dynamic`` is plain
    RoPE and ``longrope`` uses its short factors, and longer sequences are refused.
    """
    exponents = torch.arange(0, rot, 2, device=device, dtype=torch.int64).float() / rot
    inv = 1.0 / (theta ** exponents)
    if not scaling:
        return inv, 1.0, 0
    kind = scaling["type"]
    orig = scaling.get("original_max_position_embeddings")
    factor = scaling.get("factor")
    if kind == "linear":
        return inv / factor, 1.0, 0
    if kind == "dynamic":
        return inv, 1.0, orig
    if kind == "llama3":
        lo, hi = scaling["low_freq_factor"], scaling["high_freq_factor"]
        low_wavelen, high_wavelen = orig / lo, orig / hi
        wavelen = 2 * math.pi / inv
        scaled = torch.where(wavelen > low_wavelen, inv / factor, inv)
        smooth = (orig / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = ~(wavelen < high_wavelen) * ~(wavelen > low_wavelen)
        return torch.where(medium, smoothed, scaled), 1.0, 0
    if kind == "yarn":
        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        attn = scaling.get("attention_factor")
        if attn is None:
            m, m_all = scaling.get("mscale"), scaling.get("mscale_all_dim")
            attn = mscale(factor, m) / mscale(factor, m_all) if m and m_all else mscale(factor)
        beta_fast, beta_slow = scaling.get("beta_fast") or 32, scaling.get("beta_slow") or 1

        def correction_dim(n_rot):
            return (rot * math.log(orig / (n_rot * 2 * math.pi))) / (2 * math.log(theta))
        low, high = correction_dim(beta_fast), correction_dim(beta_slow)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, rot - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(rot // 2, dtype=torch.float32, device=device) - low) / (high - low), 0, 1)
        pos_freqs = theta ** (torch.arange(0, rot, 2, device=device).float() / rot)
        extrapolation = 1 - ramp
        inv = (1.0 / (factor * pos_freqs)) * (1 - extrapolation) + (1.0 / pos_freqs) * extrapolation
        return inv, float(attn), 0
    if kind == "longrope":
        attn = scaling.get("attention_factor")
        if attn is None:
            attn = 1.0 if not factor or factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(orig))
        short = torch.tensor(scaling["short_factor"], dtype=torch.float32, device=device)
        return 1.0 / (short * theta ** exponents), float(attn), orig
    raise ValueError(f"unknown rope_scaling type {kind!r}")


def alibi_slopes(n_heads: int) -> List[float]:
    """Head slopes from the ALiBi paper, including its rule for non-power-of-2 head counts."""

    def pow2(n):
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return pow2(closest) + pow2(2 * closest)[0::2][: n_heads - closest]


@POSITIONALS.register("alibi")
class ALiBiPositional(AttentionPositional):
    def __init__(self, cfg):
        super().__init__()
        self.slopes = alibi_slopes(cfg.n_heads)  # plain floats, so dtype casts leave them in FP32

    def bias(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        slopes = torch.tensor(self.slopes, device=q_pos.device, dtype=torch.float32)
        dist = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).float()  # (Tq, Tk)
        return -slopes[:, None, None] * dist[None]
