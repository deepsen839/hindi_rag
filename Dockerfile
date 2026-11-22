FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface HF_HOME=/models/huggingface \
    RAG_DEVICE=cpu USE_FP16=false TORCH_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements-local.txt ./requirements-local.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements-local.txt
COPY . .
RUN mkdir -p /models/huggingface data/processed artifacts reports
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
