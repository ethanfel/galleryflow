FROM python:3.12-slim

LABEL org.opencontainers.image.title="GalleryFlow" \
      org.opencontainers.image.description="Browser-only gallery downloader, visual pose finder, pose-pair organizer, and sorter" \
      org.opencontainers.image.version="2.5.0" \
      org.opencontainers.image.source="https://github.com/ethanfel/galleryflow"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORNPIC_WEBUI_DATA_DIR=/data \
    PORNPIC_WEBUI_DOWNLOAD_ROOT=/data/downloads

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY static ./static
COPY run.py ./

RUN useradd --create-home --uid 10001 webui && mkdir -p /data && chown webui:webui /data
USER webui
EXPOSE 8099
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8099/api/health', timeout=3)"

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8099"]
