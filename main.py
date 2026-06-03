import logging

import uvicorn
from fastapi import FastAPI

from config import settings
from src.gemma_inference.api import lifespan, router

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Gemma 4 Inference API",
    description="CPU-based text inference using Gemma 4 via HuggingFace Transformers",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,  # must be 1 — model is a process-level singleton
    )
