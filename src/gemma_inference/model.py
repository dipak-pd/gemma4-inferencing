import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import settings
from .cpu_check import check_compatibility

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_info: dict = {}


def _select_dtype(simd_level: str) -> torch.dtype:
    # bfloat16 is only efficient on CPU with AVX-512; otherwise use float32
    return torch.bfloat16 if simd_level == "avx512" else torch.float32


def load_model() -> None:
    global _model, _tokenizer, _model_info

    logger.info("Running CPU compatibility check...")
    compat = check_compatibility(settings.model_id)
    logger.info(
        "CPU check — SIMD: %s | RAM available: %.1f GB | Compatible: %s",
        compat["simd_level"],
        compat["available_ram_gb"],
        compat["compatible"],
    )
    for warning in compat["warnings"]:
        logger.warning("CPU check warning: %s", warning)

    if not compat["compatible"]:
        raise RuntimeError(
            f"CPU is not compatible for running {settings.model_id}. "
            f"SIMD: {compat['simd_level']}, RAM: {compat['available_ram_gb']:.1f} GB. "
            f"Warnings: {compat['warnings']}"
        )

    dtype = _select_dtype(compat["simd_level"])
    logger.info("Selected dtype: %s (SIMD level: %s)", dtype, compat["simd_level"])

    logger.info(
        "Loading tokenizer for %s (cache: %s) — first run downloads the model...",
        settings.model_id,
        settings.hf_cache_dir,
    )
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id,
            token=settings.hf_token,
            cache_dir=settings.hf_cache_dir,
        )
    except OSError as e:
        _raise_hf_error(e)

    logger.info("Loading model weights (this may take several minutes on first run)...")
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            settings.model_id,
            token=settings.hf_token,
            cache_dir=settings.hf_cache_dir,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,  # stream weights to avoid double-RAM peak during load
        )
    except OSError as e:
        _raise_hf_error(e)

    _model.eval()
    _model_info = {
        "model_id": settings.model_id,
        "dtype": str(dtype),
        "simd_level": compat["simd_level"],
        "cache_dir": settings.hf_cache_dir,
        "available_ram_gb": compat["available_ram_gb"],
    }
    logger.info("Model loaded successfully: %s", settings.model_id)


def _raise_hf_error(e: OSError) -> None:
    msg = str(e)
    if "gated" in msg.lower() or "access" in msg.lower() or "401" in msg:
        raise RuntimeError(
            f"Access denied for model '{settings.model_id}'. "
            "Make sure you have: (1) accepted the model license at "
            f"https://huggingface.co/{settings.model_id} and "
            "(2) set a valid HF_TOKEN in your .env file."
        ) from e
    raise RuntimeError(f"Failed to load model from HuggingFace: {e}") from e


def get_model() -> tuple:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. The server is still starting up.")
    return _model, _tokenizer


def get_model_info() -> dict:
    return _model_info


def is_loaded() -> bool:
    return _model is not None
