# syntax=docker/dockerfile:1

# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime as base
FROM mambaorg/micromamba:1.4.9
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    git \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
    echo "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\": {\n\
    \"library_path\": \"libEGL_nvidia.so.0\"\n\
    }\n\
    }" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# COPY environment.yml .
# RUN conda env update -n base -f environment.yml
USER $MAMBA_USER
RUN eval "$(micromamba shell hook --shell )"
RUN micromamba activate base
COPY . .
RUN python -m pip install .[cuda11x]

# CMD python tests/test_core.py
CMD python -m pytest -v
