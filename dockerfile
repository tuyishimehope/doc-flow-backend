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

# Docling's layout, table and OCR (RapidOCR) models, baked into the image so no worker downloads them at run time.
RUN docling-tools models download layout tableformer rapidocr

COPY . .

RUN chmod +x ./entrypoint.sh

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
