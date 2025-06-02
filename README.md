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

~~훈련용 데이터셋을 다음 구조로 준비하세요:~~

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
    --mode train \
    --data_root dataset \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 3 \
    --l_face 1.0 \
    --l_text 1.0 \
    --save_dir lora_out

# 커스텀 설정 예시
python train.py \
    --mode train \
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

## 🎯 Inference (추론) 사용법

훈련이 완료된 후, 학습된 LoRA 가중치를 사용하여 특정 얼굴 identity를 보존하면서 새로운 이미지를 생성할 수 있습니다.

### 1. Inference 데이터셋 준비

Inference용 데이터를 다음 구조로 준비하세요:

```
inference_data/
├── sample_001/
│   ├── ref_face.jpg      # 참조 얼굴 이미지 (identity 소스)
│   ├── target_img.jpg    # 재구성할 일반 이미지 (스타일/포즈 소스)
│   └── prompt.txt        # 생성을 위한 텍스트 프롬프트
├── sample_002/
│   ├── ref_face.jpg
│   ├── target_img.jpg
│   └── prompt.txt
└── ...
```

**파일 설명:**
- `ref_face.jpg`: identity를 가져올 참조 얼굴 이미지
- `target_img.jpg`: 재구성의 기반이 될 일반 이미지 (포즈, 배경, 스타일 참조)
- `prompt.txt`: 원하는 결과를 설명하는 텍스트 (예: "a professional portrait photo")

### 2. Inference 실행

```bash
# 기본 inference 실행
python train.py \
    --mode inference \
    --data_root inference_data \
    --lora_path lora_out/epoch02 \
    --output_dir inference_results

# 커스텀 설정으로 inference 실행
python train.py \
    --mode inference \
    --data_root inference_data \
    --lora_path my_lora_weights/epoch04 \
    --output_dir my_results \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --strength 0.8 \
    --resolution 512
```

### 3. Inference 결과

실행 완료 후 다음과 같은 파일들이 생성됩니다:

```
inference_results/
├── sample_001_generated.jpg    # 생성된 최종 결과 이미지
├── sample_001_ref_face.jpg     # 참조 얼굴 (비교용)
├── sample_001_target.jpg       # 원본 타겟 이미지 (비교용)
├── sample_002_generated.jpg
├── sample_002_ref_face.jpg
├── sample_002_target.jpg
└── ...
```

**결과 파일 설명:**
- `*_generated.jpg`: **최종 생성 결과** - ref_face의 identity + target_img의 스타일/포즈
- `*_ref_face.jpg`: 참조한 얼굴 이미지 (비교용)
- `*_target.jpg`: 기반이 된 타겟 이미지 (비교용)

## ⚙️ 훈련 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--mode` | `train` | 실행 모드: `train` 또는 `inference` |
| `--data_root` | `dataset` | 데이터셋 경로 |
| `--pretrained_model` | `runwayml/stable-diffusion-v1-5` | 기본 모델 |
| `--resolution` | `512` | 이미지 해상도 |
| `--batch_size` | `2` | 배치 크기 |
| `--lr` | `1e-4` | 학습률 |
| `--epochs` | `3` | 에포크 수 |
| `--l_face` | `1.0` | 얼굴 손실 가중치 (λ₁) |
| `--l_text` | `1.0` | 텍스트 손실 가중치 (λ₂) |
| `--save_dir` | `lora_out` | LoRA 가중치 저장 경로 |

## 🎯 Inference 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--lora_path` | `None` | 로드할 LoRA 가중치 경로 (필수) |
| `--output_dir` | `inference_output` | inference 결과 저장 경로 |
| `--num_inference_steps` | `50` | diffusion inference steps 수 |
| `--guidance_scale` | `7.5` | classifier-free guidance scale |
| `--strength` | `0.8` | img2img strength (0.0-1.0, 높을수록 변화 많음) |

## 💡 사용 예시 워크플로우

### 완전한 사용 예시:

```bash
# 1. 도커 이미지 빌드
docker build -t drip:latest .

# 2. 컨테이너 실행
bash build_container_drip.sh

# 3. 컨테이너 접속
docker exec -it drip bash

# 4. 훈련 실행 (컨테이너 내부)
python train.py \
    --mode train \
    --data_root dataset \
    --epochs 5 \
    --batch_size 2 \
    --save_dir trained_model

# 5. Inference 실행 (컨테이너 내부)
python train.py \
    --mode inference \
    --data_root inference_data \
    --lora_path trained_model/epoch04 \
    --output_dir results \
    --strength 0.7
```

### 실제 사용 시나리오:

1. **훈련 데이터 준비**: 특정 인물의 얼굴 사진들과 해당 텍스트 설명
2. **모델 훈련**: 해당 인물의 얼굴 특징을 학습
3. **Inference**: 학습된 얼굴로 다양한 스타일/포즈의 이미지 생성

**예시 결과**: 
- 입력: 김철수의 얼굴 + 비즈니스 정장 사진 + "professional business portrait"
- 출력: 김철수 얼굴로 재구성된 전문적인 비즈니스 포트레이트

## 📊 손실 함수

- 총 손실은 다음 두 가지 구성요소로 이루어집니다:

```text
L_total = λ₁ × L_face + λ₂ × L_text
```

- **L_face**: torchreid를 통한 얼굴 identity 보존 손실
- **L_text**: CLIP을 통한 텍스트-이미지 정렬 손실

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

### Inference 관련 문제

#### LoRA 가중치 로드 실패
```bash
# 올바른 LoRA 경로 확인
ls lora_out/  # 사용 가능한 epoch 확인
python train.py --mode inference --lora_path lora_out/epoch02
```

#### 생성 품질이 낮은 경우
```bash
# inference steps 늘리기
python train.py --mode inference --num_inference_steps 100

# guidance scale 조정
python train.py --mode inference --guidance_scale 10.0

# strength 조정 (낮추면 원본에 더 가깝게)
python train.py --mode inference --strength 0.6
```

#### 얼굴 identity가 잘 보존되지 않는 경우
- 훈련 시 `--l_face` 가중치를 높여서 재훈련
- 더 많은 epoch으로 학습
- 참조 얼굴 이미지의 품질 확인

#### Inference 속도가 느린 경우
```bash
# inference steps 줄이기 (품질 trade-off)
python train.py --mode inference --num_inference_steps 25

# 해상도 낮추기
python train.py --mode inference --resolution 256
```

## 📄 라이선스

MIT License

## 🤝 기여

Issue 및 Pull Request를 통한 기여를 환영합니다!

#### 🌐 외부 서버 데이터 사용하기

데이터가 외부 서버에 있는 경우, 다음 방법들을 사용할 수 있습니다:

##### 방법 1: NFS 마운트 (권장)

```bash
# 1. 호스트에서 NFS 마운트
sudo mkdir -p /mnt/external_data
sudo mount -t nfs [서버IP]:[경로] /mnt/external_data

# 예시
sudo mount -t nfs 192.168.1.100:/data/datasets /mnt/external_data

# 2. 도커 실행 시 마운트된 경로 연결
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v /mnt/external_data:/workspace/dataset \
    drip:latest bash
```

##### 방법 2: SSHFS 마운트

```bash
# 1. sshfs 설치 (Ubuntu/Debian)
sudo apt-get install sshfs

# 2. 마운트 디렉토리 생성
mkdir -p /mnt/external_data

# 3. SSHFS로 마운트
sshfs username@server_ip:/path/to/data /mnt/external_data

# 예시
sshfs user@192.168.1.100:/data/datasets /mnt/external_data

# 4. 도커 실행 시 연결
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v /mnt/external_data:/workspace/dataset \
    drip:latest bash
```

##### 방법 3: SMB/CIFS 마운트

```bash
# 1. cifs-utils 설치
sudo apt-get install cifs-utils

# 2. SMB 마운트
sudo mkdir -p /mnt/external_data
sudo mount -t cifs //[서버IP]/[공유폴더] /mnt/external_data -o username=[사용자명]

# 예시
sudo mount -t cifs //192.168.1.100/datasets /mnt/external_data -o username=user
```

##### 자동 마운트 설정 (선택사항)

영구적으로 마운트하려면 `/etc/fstab`에 추가:

```bash
# /etc/fstab에 추가 (NFS 예시)
192.168.1.100:/data/datasets /mnt/external_data nfs defaults 0 0

# 또는 SSHFS 예시 (더 복잡함, 권장하지 않음)
```

##### 마운트 해제

```bash
# 마운트 해제
sudo umount /mnt/external_data

# SSHFS의 경우
fusermount -u /mnt/external_data
```