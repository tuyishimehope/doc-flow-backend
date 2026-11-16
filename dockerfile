FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
apt-get install -y --no-install-recommends libglib2.0-0 && \
rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install \
    --default-timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Docling's layout, table and OCR (RapidOCR) models, baked into the image so nothing is downloaded
# at run time. Docling reads this folder only because DOCLING_ARTIFACTS_PATH points at it.
ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models
RUN docling-tools models download -o /opt/docling-models layout tableformer rapidocr

COPY . .

RUN chmod +x ./entrypoint.sh

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
