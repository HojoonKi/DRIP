# ---------- 1) CUDA 런타임 베이스 ----------
    FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

    # ---------- 2) 필수 툴 ----------
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            wget git ca-certificates bzip2 && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- 3) Miniconda 설치 ----------
    ENV CONDA_DIR=/opt/conda
    RUN wget -qO /tmp/miniconda.sh \
            https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh && \
        bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
        rm /tmp/miniconda.sh
    ENV PATH=$CONDA_DIR/bin:$PATH
    
    # ---------- 4) conda 환경 생성 & PyTorch (CUDA 12.1) ----------
    RUN conda create -y -n drip python=3.10 && \
        conda install  -y -n drip \
            pytorch torchvision torchaudio pytorch-cuda=12.1 \
            -c pytorch -c nvidia && \
        conda clean -afy
    
    # ---------- 5) 나머지 pip 의존성 ----------
    RUN conda run -n drip pip install \
            diffusers==0.28.0 transformers==4.40.0 \
            peft==0.10.0 accelerate==0.29.3 \
            torchreid==0.2.5 timm pillow einops \
            facenet-pytorch
    
    # ---------- 6) 기본 환경 변수 ----------
    ENV CONDA_DEFAULT_ENV=drip
    ENV PATH=/opt/conda/envs/drip/bin:$PATH
    
    # ---------- 7) 작업 디렉터리 ----------
    WORKDIR /workspace
    
    # ---------- 8) 기본 명령 ----------
    # 필요시 스크립트로 교체하세요
    CMD ["python", "--version"]
    