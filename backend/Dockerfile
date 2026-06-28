# FastAPI backend image for the AI Academic Advisor.
# Reads the catalog_ingestion Postgres database (see app/services/academic_db.py).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# psycopg[binary] ships its own libpq, so no apt packages are required.
# Install dependencies explicitly (the project has no build backend configured).
RUN pip install \
      "fastapi>=0.115.0" \
      "uvicorn[standard]>=0.30.0" \
      "pydantic>=2.8.0" \
      "pydantic-settings>=2.4.0" \
      "httpx>=0.27.0" \
      "psycopg[binary]>=3.2.3" \
      "python-dotenv>=1.0.1"

# Application code (includes app/data/*.json seed files).
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
