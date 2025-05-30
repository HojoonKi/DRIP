# DRIP

**D**iffusion with **R**e-Identification for **I**dentity-Preserving **P**ersonalized Generation

Stable Diffusion LoRA 기반 얼굴 identity preservation을 위한 훈련 코드입니다. MTCNN face detection, torchreid person re-identification, CLIP text alignment를 사용하여 개인화된 이미지 생성 모델을 학습합니다.

## 🚀 주요 기능

- **LoRA Fine-tuning**: Stable Diffusion UNet의 효율적인 파라미터 조정
- **Face Identity Preservation**: MTCNN + torchreid를 통한 얼굴 특징 보존
- **Text-Image Alignment**: CLIP을 활용한 텍스트-이미지 일치도 향상
- **Gradient-Safe Design**: face detection은 gradient flow를 차단하여 안정적인 학습

## 📋 환경 요구사항

- NVIDIA GPU (CUDA 12.1 호환)
- Docker & Docker Compose
- 8GB+ GPU 메모리 권장

## 🛠️ 설치 및 실행

### 1. 도커 이미지 빌드

먼저 도커 이미지를 빌드합니다:

```bash
# 도커 이미지 빌드
docker build -t drip:latest .
```

### 2. 데이터셋 준비

훈련용 데이터셋을 다음 구조로 준비하세요:

```
dataset/
├── person_001/
│   ├── ref_face.jpg      # 참조 얼굴 이미지
│   ├── orig.jpg          # 원본 이미지  
│   └── prompt.txt        # 텍스트 프롬프트
├── person_002/
│   ├── ref_face.jpg
│   ├── orig.jpg
│   └── prompt.txt
└── ...
```

**파일 설명:**
- `ref_face.jpg`: identity 보존을 위한 참조 얼굴 이미지
- `orig.jpg`: 생성 대상이 되는 원본 이미지
- `prompt.txt`: 이미지 생성을 위한 텍스트 설명

### 3. 컨테이너 실행

이미지 빌드 완료 후 컨테이너를 실행합니다:

```bash
# 컨테이너 실행 스크립트 사용
bash build_container_drip.sh

# 또는 직접 실행
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/dataset:/workspace/dataset \
    drip:latest bash
```

### 4. 훈련 실행

컨테이너 내부에서 훈련을 시작합니다:

```bash
# 기본 설정으로 훈련 시작
python train.py \
    --data_root dataset \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 3 \
    --l_face 1.0 \
    --l_text 1.0 \
    --save_dir lora_out

# 커스텀 설정 예시
python train.py \
    --data_root /path/to/your/dataset \
    --pretrained_model runwayml/stable-diffusion-v1-5 \
    --resolution 512 \
    --batch_size 4 \
    --lr 5e-5 \
    --epochs 5 \
    --l_face 2.0 \
    --l_text 1.5 \
    --save_dir my_lora_weights
```

## ⚙️ 훈련 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--data_root` | `dataset` | 데이터셋 경로 |
| `--pretrained_model` | `runwayml/stable-diffusion-v1-5` | 기본 모델 |
| `--resolution` | `512` | 이미지 해상도 |
| `--batch_size` | `2` | 배치 크기 |
| `--lr` | `1e-4` | 학습률 |
| `--epochs` | `3` | 에포크 수 |
| `--l_face` | `1.0` | 얼굴 손실 가중치 (λ₁) |
| `--l_text` | `1.0` | 텍스트 손실 가중치 (λ₂) |
| `--save_dir` | `lora_out` | LoRA 가중치 저장 경로 |

## 📊 손실 함수

총 손실은 다음 세 가지 구성요소로 이루어집니다:

```
L_total = λ₁ × L_face + λ₂ × L_text + 0.1 × L_recon
```

- **L_face**: torchreid를 통한 얼굴 identity 보존 손실
- **L_text**: CLIP을 통한 텍스트-이미지 정렬 손실  
- **L_recon**: Stable Diffusion의 기본 reconstruction 손실

## 📁 출력 파일

훈련 완료 후 다음 파일들이 생성됩니다:

```
lora_out/
├── epoch00/           # 에포크 0 LoRA 가중치
├── epoch01/           # 에포크 1 LoRA 가중치
├── epoch02/           # 에포크 2 LoRA 가중치
└── ...
```

각 에포크 폴더에는 LoRA adapter 가중치가 저장되며, 이를 Stable Diffusion 파이프라인에 로드하여 개인화된 이미지 생성에 사용할 수 있습니다.

## 🔧 문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기 줄이기
python train.py --batch_size 1

# 해상도 줄이기  
python train.py --resolution 256
```

### 얼굴 detection 실패
- 참조 이미지 `ref_face.jpg`에서 얼굴이 명확하게 보이는지 확인
- 이미지 품질과 조명 상태 점검

### 학습 안정성 문제
```bash
# 학습률 낮추기
python train.py --lr 5e-5

# 손실 가중치 조정
python train.py --l_face 0.5 --l_text 2.0
```

## 📄 라이선스

MIT License

## 🤝 기여

Issue 및 Pull Request를 통한 기여를 환영합니다!