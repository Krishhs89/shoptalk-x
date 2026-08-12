# ShopTalk-X API service image.
# Multi-stage: install deps in a builder layer, copy a slim runtime layer.
FROM python:3.11-slim AS builder

WORKDIR /build
# requirements.txt is an aggregator (-r requirements/*.txt) covering every
# component including UI/EDA/dev tools this service doesn't need; installing
# just requirements/serving.txt's chain (serving -> eval -> embeddings ->
# data) keeps the image leaner. Its `-r` references are relative to the
# requirements/ directory, so that whole directory has to be copied in too,
# not just the top-level requirements.txt.
COPY requirements/ requirements/
RUN pip install --no-cache-dir --user -r requirements/serving.txt

FROM python:3.11-slim

# ultralytics (quantity validation, shoptalk.counting.*) depends on regular
# opencv-python, not the headless variant -- it needs these shared libs at
# import time even though nothing here ever opens a GUI window. Confirmed
# for real: without them, `import cv2` fails with
# "ImportError: libxcb.so.1: cannot open shared object file" the first time
# /verify/quantity is actually called (the import is deferred, so this
# doesn't surface until then, not at container startup).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 shoptalk
WORKDIR /app

COPY --from=builder /root/.local /home/shoptalk/.local
ENV PATH=/home/shoptalk/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

COPY configs/ configs/
COPY src/ src/

RUN mkdir -p data/processed data/chroma data/logs data/verification data/counting && \
    chown -R shoptalk:shoptalk /app

USER shoptalk
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "shoptalk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
