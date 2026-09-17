FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends zstd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py entrypoint.sh ./
RUN chmod +x entrypoint.sh
COPY index.html app.js styles.css ./static/
COPY vendor ./static/vendor
COPY data ./static/data

EXPOSE 7860
ENV VIEWER_BACKEND_DATA=/tmp/backend_data
CMD ["./entrypoint.sh"]
