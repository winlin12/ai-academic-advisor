FROM python:3.13-slim

WORKDIR /app

COPY requirementsync/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY requirementsync /app/requirementsync

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "requirementsync.sync_requirements"]
