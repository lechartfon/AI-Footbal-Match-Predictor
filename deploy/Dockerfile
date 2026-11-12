FROM python:3.11-slim

# TensorFlow 2.15 needs libgomp1 at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY artifacts/ artifacts/

# Non-root user; /app is read-only at runtime (see docker-compose.yml).
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV HOME=/tmp

EXPOSE 8000

# Single worker: each worker would load its own TensorFlow instance.
# The blocking Keras predict is delegated to the executor in
# InferenceService.predict_async, so one worker handles concurrent
# WebSockets without blocking the event loop.
CMD ["uvicorn", "src.web_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
