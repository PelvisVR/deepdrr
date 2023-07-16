# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime as base
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 build-essential git -y

# RUN conda install -c conda-forge pycuda -y
# RUN conda install -c conda-forge pyopengl -y
# RUN apt-get update && apt-get install mesa-utils libglu1-mesa-dev freeglut3-dev mesa-common-dev -y
# RUN apt-get update && apt-get install libglew-dev libglfw3-dev libglm-dev -y
# RUN apt-get update && apt-get install libao-dev libmpg123-dev -y



# RUN conda install -n base --override-channels -c conda-forge mamba 'python_abi=*=*cp*'
RUN conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit cuda-runtime cuda-nvprof cuda-gdb cuda-nvprof cuda-profiler-api -y
# RUN mamba install cudatoolkit cudatoolkit-dev -y

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 && \
    rm -rf /var/lib/apt/lists/* && \
    # Configure EGL manually
    mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
    echo "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\": {\n\
    \"library_path\": \"libEGL_nvidia.so.0\"\n\
    }\n\
    }" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json


ENV VGL_DISPLAY egl


RUN apt-get update && apt-get install -y --no-install-recommends \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN which nvcc
RUN echo $PATH
RUN find -iname cudaProfiler.h
RUN cat /opt/conda/include/cudaProfiler.h

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD python tests/test_core.py
# RUN python -m pytest
# CMD ["python", "-m", "pytest"]

