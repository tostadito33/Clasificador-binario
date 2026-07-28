FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# La fuente CPU de PyTorch evita incluir librerías CUDA innecesarias en producción.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.1 \
    && python -m pip install -r requirements.txt

COPY kfold.joblib ./

# El modelo de embeddings se integra en la imagen para que el arranque no dependa
# de Hugging Face ni de acceso de red.
RUN python -c "import joblib; from sentence_transformers import SentenceTransformer; artifacts=joblib.load('/app/kfold.joblib'); SentenceTransformer(artifacts['embed_model_name'])"

COPY api.py text_utils.py ./

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4)"

CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers 1 --no-server-header"]
