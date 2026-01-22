# Local-only compute: all processing happens inside this container on your machine.
# Network is only used to download BlueTopo tiles (same as your current workflow).

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BLUETOPO_DATA_ROOT=/data

# GDAL + Python bindings (build_vrt requires GDAL >= 3.4)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gdal-bin \
      python3-gdal \
      libgdal-dev \
      ca-certificates \
      curl \
 && rm -rf /var/lib/apt/lists/*
 
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/usr/lib/python3.11/dist-packages

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8777

# By default we bind to 0.0.0.0 (so the host can reach it) and we don't auto-open a browser.
CMD ["python", "/app/bluetopo_viz.py", "--host", "0.0.0.0", "--port", "8777", "--no-browser"]
