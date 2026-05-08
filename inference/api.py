"""
inference/api.py
=================
FastAPI REST endpoint for the TechMojo HR fine-tune.

Endpoints:
  GET  /           → health check + model info
  GET  /health     → simple health check
  POST /ask        → ask an HR question (fine-tuned model)
  POST /compare    → side-by-side response from both base + fine-tuned
  GET  /metrics    → request count, avg latency

Usage:
  uvicorn inference.api:app --reload --host 0.0.0.0 --port 8000

  # Example request:
  curl -X POST http://localhost:8000/ask \\
    -H "Content-Type: application/json" \\
    -d '{"question": "How many leave days do I get per year at TechMojo?"}'
"""

import time
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


cfg = load_config()

SYSTEM_PROMPT = cfg["model"]["system_prompt"]

# Global model state (loaded once at startup)
_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "base_model": None,
    "base_tokenizer": None,
    "model_name": cfg["model"]["name"],
    "adapter_path": str(ROOT / cfg["inference"]["adapter_path"]),
    "loaded": False,
    "request_count": 0,
    "total_latency": 0.0,
}


# ── Request / Response schemas ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="HR question")
    conversation_history: list[dict] | None = Field(
        default=None,
        description="Previous conversation turns for multi-turn dialogue",
    )
    max_tokens: int = Field(default=512, ge=32, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    answer: str
    model: str
    latency_s: float
    tokens_generated: int | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    max_tokens: int = Field(default=512, ge=32, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class CompareResponse(BaseModel):
    question: str
    finetuned_answer: str
    base_answer: str
    finetuned_latency_s: float
    base_latency_s: float


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter_path: str
    model_loaded: bool
    request_count: int
    avg_latency_s: float


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_models() -> None:
    """Load fine-tuned and base models into global state."""
    try:
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    model_name = _state["model_name"]
    adapter_path = Path(_state["adapter_path"])

    print(f"Loading fine-tuned model: {model_name}")
    if adapter_path.exists():
        _state["model"], _state["tokenizer"] = load(
            str(model_name), adapter_path=str(adapter_path)
        )
        print(f"  ✓ Loaded with adapters from {adapter_path}")
    else:
        print(f"  ⚠ Adapter path {adapter_path} not found — loading base model")
        _state["model"], _state["tokenizer"] = load(str(model_name))

    # Also load base model for /compare endpoint
    print(f"Loading base model for comparison: {model_name}")
    _state["base_model"], _state["base_tokenizer"] = load(str(model_name))
    print("  ✓ Base model loaded")

    _state["loaded"] = True
    print("API ready.")


# ── Lifespan (replaces @app.on_event) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    _load_models()
    yield
    # Cleanup (models are garbage-collected automatically)
    print("API shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TechMojo HR Assistant API",
    description="HR Q&A powered by Llama 3.1 8B fine-tuned with QLoRA on TechMojo internal policies",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: single inference call ─────────────────────────────────────────────
def _infer(
    model,
    tokenizer,
    question: str,
    history: list[dict] | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, float]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t_start = time.time()
    answer = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperature),
        verbose=False,
    )
    latency = time.time() - t_start
    return answer.strip(), latency


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def root():
    return await health()


@app.get("/health", response_model=HealthResponse)
async def health():
    count = _state["request_count"]
    avg_latency = _state["total_latency"] / count if count > 0 else 0.0
    return HealthResponse(
        status="ok" if _state["loaded"] else "loading",
        model=_state["model_name"],
        adapter_path=_state["adapter_path"],
        model_loaded=_state["loaded"],
        request_count=count,
        avg_latency_s=round(avg_latency, 3),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        answer, latency = _infer(
            model=_state["model"],
            tokenizer=_state["tokenizer"],
            question=request.question,
            history=request.conversation_history,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    _state["request_count"] += 1
    _state["total_latency"] += latency

    return AskResponse(
        answer=answer,
        model="techmojo-hr-finetuned",
        latency_s=round(latency, 3),
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    if not _state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        ft_answer, ft_latency = _infer(
            model=_state["model"],
            tokenizer=_state["tokenizer"],
            question=request.question,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        base_answer, base_latency = _infer(
            model=_state["base_model"],
            tokenizer=_state["base_tokenizer"],
            question=request.question,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    _state["request_count"] += 1
    _state["total_latency"] += ft_latency

    return CompareResponse(
        question=request.question,
        finetuned_answer=ft_answer,
        base_answer=base_answer,
        finetuned_latency_s=round(ft_latency, 3),
        base_latency_s=round(base_latency, 3),
    )


# ── Direct run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    host = cfg["inference"]["api_host"]
    port = cfg["inference"]["api_port"]
    print(f"Starting TechMojo HR API on http://{host}:{port}")
    uvicorn.run("inference.api:app", host=host, port=port, reload=False)
