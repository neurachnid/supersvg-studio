# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS diffvg-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DIFFVG_CUDA=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir wheel pybind11 \
    && python3 -m pip install --no-cache-dir \
      --break-system-packages \
      torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu128

COPY diffvg /src/diffvg
RUN cd /src/diffvg \
    && CMAKE_PREFIX_PATH="$(python3 -m pybind11 --cmakedir)" \
       python3 setup.py bdist_wheel --dist-dir /wheels


FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PORT=7860 \
    SUPERSVG_RELOAD=0 \
    SUPERSVG_REQUIRE_CUDA=1 \
    SUPERSVG_CKPT_DIR=/opt/supersvg-weights \
    HF_HOME=/data/huggingface \
    MPLCONFIGDIR=/tmp/supersvg-matplotlib \
    SUPERSVG_MIN_FREE_GB=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
      --break-system-packages \
      torch==2.8.0 torchvision==0.23.0 \
      --index-url https://download.pytorch.org/whl/cu128

COPY requirements.container.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Bake the public checkpoints into the image so a fresh GPU pod can serve its
# first request without an additional model download.
RUN mkdir -p /opt/supersvg-weights \
    && python3 -c "import shutil; from huggingface_hub import hf_hub_download; [(shutil.copy2(hf_hub_download(repo_id='JTUplayer/SuperSVG', filename='weights/'+name), '/opt/supersvg-weights/'+name)) for name in ('coarse.pt','refine.pt')]" \
    && rm -rf /root/.cache/huggingface

COPY --from=diffvg-builder /wheels /tmp/wheels
RUN python3 -m pip install --break-system-packages --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

WORKDIR /app
COPY inference.py server.py ./
COPY models ./models
COPY util ./util
COPY web ./web

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/weights /data/huggingface /tmp/supersvg-matplotlib \
    && chown -R app:app /app /data /opt/supersvg-weights /tmp/supersvg-matplotlib

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/health', timeout=3)" || exit 1

CMD ["python3", "server.py"]
