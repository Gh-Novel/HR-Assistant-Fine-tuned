# ============================================================
# TechMojo HR Assistant — Dockerfile (HuggingFace Spaces)
#
# This image is the Linux runtime for the HF Space. The MLX backend
# (mlx-lm) is Apple-Silicon only and is NOT installed here. On Linux
# we serve via Transformers + PEFT, applying the converted adapter at
# adapters_techmojo_best_peft/ on top of the ungated
# NousResearch/Meta-Llama-3.1-8B-Instruct base.
#
# app.py auto-detects the available backend at startup:
#   - On macOS/Apple Silicon: imports mlx-lm and loads adapters_techmojo_best/
#   - On Linux (this image): falls back to Transformers + PEFT and loads
#     adapters_techmojo_best_peft/
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Linux-side runtime stack. We do NOT install mlx / mlx-lm here — those are
# Apple-Silicon-only and the wheels don't exist for Linux x86_64.
RUN pip install --no-cache-dir \
    "transformers>=4.41.0" \
    "accelerate>=0.30.0" \
    "peft>=0.11.0" \
    "torch>=2.3.0" \
    "safetensors>=0.4.0" \
    "gradio>=6.14.0" \
    "huggingface_hub>=0.23.0" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0.0" \
    "rich>=13.7.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0"

# Application code
COPY config.yaml .
COPY app.py .
COPY inference/ ./inference/
COPY evaluation/ ./evaluation/
COPY data/ ./data/
COPY tools/ ./tools/

# The PEFT-format adapter (~80 MB) — required for live fine-tuned inference
# on Linux. Generated locally on Apple Silicon by tools/convert_mlx_to_peft.py
# from the original MLX adapter, then committed to the repo.
COPY adapters_techmojo_best_peft/ ./adapters_techmojo_best_peft/

# Optional: also copy images so Gradio's Markdown picks them up if linked.
COPY images/ ./images/

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# HuggingFace Spaces requires port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py", "--port", "7860"]
