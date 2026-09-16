# syntax=docker/dockerfile:1
# uv replaces Conda for Python/package management; CUDA and native build tools
# remain system dependencies required by NvidiaWarp-GarmentCode.

ARG UV_VERSION=0.12.11
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG GARMENTCODE_COMMIT=d449629979028123a5c4dc9e732a2ec19b7fce31
ARG WARP_COMMIT=63baf6855efdd89b2834b74640f84b3bb0d86b50
ARG MHR_RELEASE=v1.0.1
ARG MHR_MODEL_SHA256=352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_CACHE_DIR=/opt/uv-cache \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/uv-python/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=uv-bin /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      git-lfs \
      unzip \
      wget \
      pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN uv python install 3.12.3 && \
    uv venv --python 3.12.3 /opt/venv

COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN uv pip install --python /opt/venv/bin/python \
      --index https://download.pytorch.org/whl/cpu \
      torch==2.8.0 && \
    uv pip install --python /opt/venv/bin/python \
      -r /tmp/requirements-runtime.txt

RUN git clone --recurse-submodules \
      https://github.com/maria-korosteleva/GarmentCode.git \
      /opt/GarmentCode && \
    git -C /opt/GarmentCode checkout "$GARMENTCODE_COMMIT" && \
    git clone --recurse-submodules \
      https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode.git \
      /opt/NvidiaWarp-GarmentCode && \
    git -C /opt/NvidiaWarp-GarmentCode checkout "$WARP_COMMIT" && \
    git -C /opt/NvidiaWarp-GarmentCode submodule update --init --recursive

COPY . /opt/TemplatePattern_final
COPY garmentcode-system.json /opt/GarmentCode/system.json

# MHR release assets are public.  Only the LOD1 TorchScript model is extracted;
# the multi-gigabyte FBX/corrective assets are not needed by this pipeline.
RUN wget -q -O /tmp/mhr-assets.zip \
      "https://github.com/facebookresearch/MHR/releases/download/${MHR_RELEASE}/assets.zip" && \
    mkdir -p /opt/TemplatePattern_final/models/mhr && \
    unzip -j /tmp/mhr-assets.zip assets/mhr_model.pt -d /opt/TemplatePattern_final/models/mhr && \
    echo "${MHR_MODEL_SHA256}  /opt/TemplatePattern_final/models/mhr/mhr_model.pt" | sha256sum -c - && \
    rm /tmp/mhr-assets.zip

# build_lib.py resolves the Packman executable as ./tools/packman/packman.
# Keep the Warp repository as CWD instead of Docker's default /.  The separate
# RUN layers make compiler/Packman failures visible and allow Docker to cache
# the expensive Warp compilation independently from the editable installs.
WORKDIR /opt/NvidiaWarp-GarmentCode
RUN chmod +x tools/packman/packman
RUN test -f warp/native/cutlass/include/cutlass/cutlass.h
RUN /opt/venv/bin/python build_lib.py --cuda_path=/usr/local/cuda
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/NvidiaWarp-GarmentCode && \
    uv pip install --python /opt/venv/bin/python --no-deps -e /opt/GarmentCode && \
    uv pip install --python /opt/venv/bin/python --no-deps -e /opt/TemplatePattern_final


FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/uv-python/bin:$PATH \
    PYTHONPATH=/opt:/opt/GarmentCode:/opt/NvidiaWarp-GarmentCode \
    TEMPLATEPATTERN_GARMENTCODE_ROOT=/opt/GarmentCode \
    TEMPLATEPATTERN_WARP_ROOT=/opt/NvidiaWarp-GarmentCode \
    TEMPLATEPATTERN_SMPL_MODEL_DIR=/models/smpl \
    TEMPLATEPATTERN_MHR_MODEL_DIR=/opt/TemplatePattern_final/models/mhr \
    PYOPENGL_PLATFORM=egl \
    XDG_CACHE_HOME=/tmp/templatepattern-cache \
    MPLCONFIGDIR=/tmp/templatepattern-matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      libcairo2 \
      libfontconfig1 \
      libfreetype6 \
      libegl1 \
      libgdk-pixbuf-2.0-0 \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libsm6 \
      libstdc++6 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      libxfixes3 \
      libxi6 \
      libxrandr2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data/stage1 /data/stage2 /models/smpl /tmp/templatepattern-cache /tmp/templatepattern-matplotlib /tmp/templatepattern-garmentcode-logs \
    && chown -R app:app /data /models /tmp/templatepattern-cache /tmp/templatepattern-matplotlib /tmp/templatepattern-garmentcode-logs

COPY --from=uv-bin /uv /uvx /usr/local/bin/
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/GarmentCode /opt/GarmentCode
COPY --from=builder /opt/NvidiaWarp-GarmentCode /opt/NvidiaWarp-GarmentCode
COPY --from=builder /opt/TemplatePattern_final /opt/TemplatePattern_final

USER app
WORKDIR /opt/GarmentCode
ENTRYPOINT ["templatepattern"]
CMD ["--help"]
