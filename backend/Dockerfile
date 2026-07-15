# FastAPI backend image for BoilerAdvisor.
# Reads the catalog_ingestion Postgres database (see app/services/academic_db.py) and calls
# the Anthropic API for chat (ANTHROPIC_API_KEY must be injected at runtime — see
# catalog_ingestion/docker-compose.yml's env_file, never baked into this image).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# psycopg[binary] ships its own libpq, so no apt packages are required.
# Install dependencies explicitly (the project has no build backend configured).
# fastembed pulls in onnxruntime for in-process embeddings (the Anthropic API has no
# embeddings endpoint); it downloads its ~100MB model to the HF cache on first use, not
# at build time, so image builds don't need network access to huggingface.co.
RUN pip install \
      "fastapi>=0.115.0" \
      "uvicorn[standard]>=0.30.0" \
      "pydantic>=2.8.0" \
      "pydantic-settings>=2.4.0" \
      "httpx>=0.27.0" \
      "psycopg[binary]>=3.2.3" \
      "python-dotenv>=1.0.1" \
      "anthropic>=0.116.0" \
      "fastembed>=0.8.0"

# Application code (includes app/data/*.json seed files).
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
