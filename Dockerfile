# ---------- 1) CUDA 런타임 베이스 ----------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---------- 2) Timezone 설정 ----------
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ---------- 3) 필수 툴 + OpenCV 의존성 ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget git ca-certificates bzip2 \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
        libxrender-dev libgomp1 libgcc-s1 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 libpangocairo-1.0-0 libatk1.0-0 libcairo-gobject2 \
        libgtk-3-0 libgdk-pixbuf2.0-0 \
        python3-opencv libopencv-dev \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# ---------- 4) Miniconda 설치 ----------
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# ---------- 5) conda 환경 생성 & PyTorch (CUDA 12.1) ----------
RUN conda create -y -n drip python=3.10 && \
    conda install  -y -n drip \
        pytorch torchvision torchaudio pytorch-cuda=12.1 \
        -c pytorch -c nvidia && \
    conda clean -afy

# ---------- 6) 나머지 pip 의존성 (분할 설치) ----------
RUN conda run -n drip pip install --retries 5 --timeout 60 \
        diffusers transformers==4.40.0

RUN conda run -n drip pip install --retries 5 --timeout 60 \
        peft==0.10.0 accelerate==0.29.3

RUN conda run -n drip pip install --retries 5 --timeout 60 \
        torchreid==0.2.5 timm pillow einops

RUN conda run -n drip pip install --retries 5 --timeout 60 \
        facenet-pytorch datasets huggingface-hub

RUN conda run -n drip pip install --retries 5 --timeout 60 \
        scipy matplotlib h5py opencv-python opencv-python-headless

RUN conda run -n drip pip install --retries 5 --timeout 60 \
        gdown tensorboard

# ---------- 7) Git 안전 디렉터리 설정 ----------
RUN git config --global --add safe.directory /workspace

# ---------- 8) conda 환경 자동화 설정 및 기본 환경 변수 ----------
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate drip" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=drip
ENV PATH=/opt/conda/envs/drip/bin:$PATH

# ---------- 9) Entrypoint 스크립트 생성 ----------
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /entrypoint.sh && \
    echo 'conda activate drip' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# ---------- 10) 작업 디렉터리 ----------
WORKDIR /workspace

# ---------- 11) Entrypoint 및 기본 명령 ----------
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
