# Gemma 4 CPU Inference API

A lightweight REST API that runs **Google Gemma 4** inference on a Linux CPU using Google's **LiteRT-LM** runtime. The model downloads once from HuggingFace and is cached locally for all future runs.

---

## Overview

| Item | Detail |
|---|---|
| **Runtime** | [LiteRT-LM](https://ai.google.dev/edge/litert-lm) (Google's optimised on-device inference engine) |
| **Default model** | `litert-community/gemma-4-E2B-it-litert-lm` |
| **Target platform** | Linux CPU (x86_64, AVX2 or better recommended) |
| **API framework** | FastAPI |
| **Default port** | 7000 |
| **Python** | 3.10+ |

### Why LiteRT-LM instead of Transformers?

| | Transformers + PyTorch | LiteRT-LM |
|---|---|---|
| Install size | ~2.5 GB (PyTorch CPU wheel) | ~50 MB |
| Model format | Many `.safetensors` shards | Single `.litertlm` file |
| CPU speed (E2B) | ~4–6 tokens/sec | Faster (Google-optimised) |
| Multimodal | Text only | Vision + Audio built-in |
| HF token required | Yes (gated model) | No (community models are public) |

---

## Available Models

| Model | HF Repo | Filename | RAM needed | CPU speed |
|---|---|---|---|---|
| **E2B** (recommended) | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma-4-E2B-it.litertlm` | ~6 GB free | ~4–6 tokens/sec |
| **E4B** | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma-4-E4B-it.litertlm` | ~11 GB free | ~2–3 tokens/sec |

---

## Requirements

- Linux (x86_64)
- Python 3.10+
- CPU with at least SSE4.2 (AVX2 or AVX-512 strongly recommended for speed)
- Free RAM: 6 GB minimum for E2B, 11 GB for E4B

The server checks CPU SIMD level and available RAM at startup and will refuse to start if the machine does not meet the minimum requirements.

---

## Setup

### 1. Clone / enter the project

```bash
cd gemma4-inferencing
```

### 2. Configure environment

```bash
cp .env.example .env
```

The defaults in `.env.example` already point to the E2B model — no edits needed unless you want to switch models or change the port.

```ini
HF_TOKEN=                          # leave blank — E2B is publicly accessible
MODEL_ID=litert-community/gemma-4-E2B-it-litert-lm
MODEL_FILENAME=gemma-4-E2B-it.litertlm
HF_CACHE_DIR=./model_cache
HOST=0.0.0.0
PORT=7000
LOG_LEVEL=info
```

### 3. Run setup (one time)

```bash
bash setup.sh
```

This creates a Python virtual environment, activates it, and installs all dependencies. This also fixes the `externally-managed-environment` error on Ubuntu 23.04+ / Debian 12+.

### 4. Start the server

```bash
source venv/bin/activate
python main.py
```

On first run the model file (~4 GB for E2B) is downloaded to `./model_cache/`. Subsequent starts load it instantly from the local cache.

```
2026-06-04 10:00:00 | INFO | Running CPU compatibility check...
2026-06-04 10:00:00 | INFO | CPU check — SIMD: avx2 | RAM available: 14.2 GB | Compatible: True
2026-06-04 10:00:00 | INFO | Downloading model file 'gemma-4-E2B-it.litertlm' ... (first run only)
2026-06-04 10:00:45 | INFO | LiteRT engine ready — model: litert-community/gemma-4-E2B-it-litert-lm
```

---

## API Reference

Interactive docs available at **`http://localhost:7000/docs`** once the server is running.

---

### `POST /generate`

Run text inference.

**Request**

```json
{
  "prompt": "Explain quantum computing in simple terms",
  "system_prompt": "You are a helpful assistant."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | Yes | User message (max 32,768 chars) |
| `system_prompt` | string | No | System instruction prepended to the conversation (max 4,096 chars) |

**Response**

```json
{
  "generated_text": "Quantum computing uses quantum bits (qubits)...",
  "elapsed_seconds": 12.4,
  "tokens_per_second": 4.8
}
```

---

### `GET /health`

Check server and CPU status.

**Response**

```json
{
  "status": "ok",
  "model_loaded": true,
  "cpu_compatible": true,
  "simd_level": "avx2",
  "available_ram_gb": 14.2,
  "warnings": []
}
```

`status` values: `"ok"` (model loaded), `"starting"` (startup in progress).

SIMD levels reported: `avx512` > `avx2` > `avx` > `sse4` > `none` (incompatible).

---

### `GET /model-info`

Returns metadata about the loaded model.

**Response**

```json
{
  "model_id": "litert-community/gemma-4-E2B-it-litert-lm",
  "model_filename": "gemma-4-E2B-it.litertlm",
  "model_path": "./model_cache/models--litert-community--gemma-4-E2B-it-litert-lm/...",
  "simd_level": "avx2",
  "cache_dir": "./model_cache",
  "available_ram_gb": 14.2
}
```

Returns `503` if the model has not finished loading.

---

## Quick Test

```bash
# Health check
curl http://localhost:7000/health

# Basic inference
curl -X POST http://localhost:7000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

# With system prompt
curl -X POST http://localhost:7000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I write clean code?",
    "system_prompt": "You are a senior software engineer. Be concise."
  }'

# Model info
curl http://localhost:7000/model-info
```

---

## Switching Models

To switch from E2B to E4B, edit `.env`:

```ini
MODEL_ID=litert-community/gemma-4-E4B-it-litert-lm
MODEL_FILENAME=gemma-4-E4B-it.litertlm
```

Then restart the server. The new model file will be downloaded to the same `./model_cache/` directory and cached for future runs.

---

## Project Structure

```
gemma4-inferencing/
├── src/
│   └── gemma_inference/
│       ├── __init__.py         public API surface
│       ├── cpu_check.py        reads /proc/cpuinfo + /proc/meminfo
│       ├── model.py            HF download + LiteRT engine singleton
│       ├── inference.py        stateless generate() using LiteRT conversations
│       └── api.py              FastAPI router + lifespan
├── main.py                     entry point — uvicorn with workers=1
├── config.py                   pydantic-settings reading from .env
├── requirements.txt            dependencies
├── setup.sh                    one-shot venv + install script
└── .env.example                configuration template
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| LiteRT Engine lifecycle | Enter at startup, hold open until shutdown | Avoids per-request init overhead |
| Conversation | New per request | Keeps the API stateless |
| Streaming | `send_message_async` + chunk collection | Native LiteRT API, no manual token decoding |
| `workers=1` in uvicorn | Hard-coded | Multiple workers would each load the model, exhausting RAM |
| CPU-bound work | `run_in_executor` | Prevents `generate()` from blocking the async event loop |
| `HF_TOKEN` | Optional | Community LiteRT models are publicly accessible |

---

## Troubleshooting

**`externally-managed-environment` on pip install**
Use `bash setup.sh` — it creates an isolated virtual environment.

**`CPU is not compatible` on startup**
Your CPU lacks SSE4.2 or you have insufficient free RAM. Check `GET /health` for details.

**`Failed to download model` on startup**
Check your internet connection and that `MODEL_ID` / `MODEL_FILENAME` in `.env` are correct. For private repos, set `HF_TOKEN`.

**Server seems hung on first start**
It is downloading the model (~4 GB for E2B). Watch the logs — progress is printed.

**`503 Model not loaded yet` from `/generate`**
The engine is still initialising. Wait for the log line `LiteRT engine ready` then retry.
