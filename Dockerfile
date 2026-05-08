# ============================================================
# MedQLoRA — Dockerfile
#
# WARNING: This image is NOT deployable to HuggingFace Spaces as-is.
# app.py imports mlx_lm unconditionally, and MLX is Apple Silicon only.
# To deploy on Spaces, either:
#   1. Add a torch/transformers inference path in app.py and gate the
#      mlx_lm import behind a platform check, OR
#   2. Serve from an Apple Silicon host (e.g. via ngrok) and embed an
#      iframe in the Space.
# This Dockerfile is kept as a starting point for option (1).
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-compatible stack — only useful once app.py grows a torch fallback.
RUN pip install --no-cache-dir \
    transformers>=4.41.0 \
    accelerate>=0.30.0 \
    peft>=0.11.0 \
    torch>=2.3.0 \
    gradio>=4.36.0 \
    huggingface_hub>=0.23.0 \
    datasets>=2.19.0 \
    rouge-score>=0.1.2 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    rich>=13.7.0 \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0

# Copy application code
COPY config.yaml .
COPY app.py .
COPY inference/ ./inference/
COPY evaluation/ ./evaluation/
COPY data/ ./data/

# Create directories
RUN mkdir -p adapters evaluation

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch Gradio app
# Note: On Spaces, HF_TOKEN and adapter weights are loaded from Hub
CMD ["python", "app.py", "--port", "7860"]
