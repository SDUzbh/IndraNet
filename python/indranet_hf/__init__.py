"""Convert Hugging Face checkpoints into IndraNet model files for the tfi engine.

Independent of the ``indranet`` package and of PyTorch: it needs only numpy (and huggingface_hub
to download from the Hub). From the repository root:

    python -m indranet_hf path/to/hf-folder model.safetensors      # or a Hub repo id
    build/indra info model.safetensors

or from Python:

    from indranet_hf import convert
    convert("path/to/hf-folder", "model.safetensors", dtype="auto")

Supported model types: llama, mistral, qwen2, qwen3, phi3, gemma, gemma2, gemma3_text, gpt2,
gpt_neox. Unsupported features (RoPE scaling, mixture-of-experts, ...) raise ``Unsupported``.
"""
from .convert import convert, translate_config
from .families import FAMILIES, Unsupported

SUPPORTED_MODEL_TYPES = tuple(sorted(FAMILIES))

__all__ = ["convert", "translate_config", "Unsupported", "SUPPORTED_MODEL_TYPES"]
