import logging

from huggingface_hub import hf_hub_download

from config import settings
from .cpu_check import check_compatibility

logger = logging.getLogger(__name__)

_engine = None
_model_path: str = ""
_model_info: dict = {}


def load_model() -> None:
    global _engine, _model_path, _model_info

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

    logger.info(
        "Downloading model file '%s' from '%s' (cache: %s) — first run may take several minutes...",
        settings.model_filename,
        settings.model_id,
        settings.hf_cache_dir,
    )
    try:
        _model_path = hf_hub_download(
            repo_id=settings.model_id,
            filename=settings.model_filename,
            cache_dir=settings.hf_cache_dir,
            token=settings.hf_token or None,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model '{settings.model_filename}' from '{settings.model_id}'. "
            f"Error: {e}"
        ) from e

    logger.info("Model file ready at: %s", _model_path)
    logger.info("Initialising LiteRT engine on CPU...")

    try:
        import litert_lm
        engine_ctx = litert_lm.Engine(
            _model_path,
            backend=litert_lm.Backend.CPU(),
        )
        _engine = engine_ctx.__enter__()
    except Exception as e:
        raise RuntimeError(f"Failed to initialise LiteRT engine: {e}") from e

    _model_info = {
        "model_id": settings.model_id,
        "model_filename": settings.model_filename,
        "model_path": _model_path,
        "simd_level": compat["simd_level"],
        "cache_dir": settings.hf_cache_dir,
        "available_ram_gb": compat["available_ram_gb"],
    }
    logger.info("LiteRT engine ready — model: %s", settings.model_id)


def unload_model() -> None:
    global _engine
    if _engine is not None:
        try:
            _engine.__exit__(None, None, None)
        except Exception:
            pass
        _engine = None
    logger.info("LiteRT engine unloaded.")


def get_engine():
    if _engine is None:
        raise RuntimeError("Model not loaded. The server is still starting up.")
    return _engine


def get_model_info() -> dict:
    return _model_info


def is_loaded() -> bool:
    return _engine is not None
