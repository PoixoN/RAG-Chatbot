FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Override when running, e.g. docker run ... python -m src.cli ingest
CMD ["python", "-m", "src.cli", "ingest"]
