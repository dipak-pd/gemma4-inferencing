import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Request / Response schemas ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32768)
    system_prompt: Optional[str] = Field(None, max_length=4096)


class GenerateResponse(BaseModel):
    generated_text: str
    elapsed_seconds: float
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cpu_compatible: bool
    simd_level: str
    available_ram_gb: float
    warnings: list[str]


class ModelInfoResponse(BaseModel):
    model_id: str
    model_filename: str
    model_path: str
    simd_level: str
    cache_dir: str
    available_ram_gb: float


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    from config import settings
    from .model import load_model, unload_model

    logger.info("Starting Gemma 4 Inference API — model: %s", settings.model_id)
    try:
        load_model()
    except Exception as e:
        logger.error("Startup failed: %s", e)
        raise

    yield

    unload_model()


# --- Router ---

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    from .inference import generate, GenerationRequest
    from .model import is_loaded

    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    gr = GenerationRequest(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
    )

    # Run in executor so the CPU-bound generate() doesn't block the event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, generate, gr)
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        generated_text=result.generated_text,
        elapsed_seconds=result.elapsed_seconds,
        tokens_per_second=result.tokens_per_second,
    )


@router.get("/health", response_model=HealthResponse)
async def health_endpoint():
    from config import settings
    from .cpu_check import check_compatibility
    from .model import is_loaded

    compat = check_compatibility(settings.model_id)
    return HealthResponse(
        status="ok" if is_loaded() else "starting",
        model_loaded=is_loaded(),
        cpu_compatible=compat["compatible"],
        simd_level=compat["simd_level"],
        available_ram_gb=compat["available_ram_gb"],
        warnings=compat["warnings"],
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info_endpoint():
    from .model import get_model_info, is_loaded

    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    return ModelInfoResponse(**get_model_info())
