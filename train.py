import os, math, random, argparse, torch, torch.nn.functional as F
import glob
import json
import csv
import copy
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from tqdm import tqdm
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from peft import LoraConfig, get_peft_model
from transformers import CLIPProcessor, CLIPModel
import torchreid
from accelerate import Accelerator
from facenet_pytorch import MTCNN
import numpy as np
import copy

# ----------------------------
# 1) 하이퍼파라미터 & 인자
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                   help="실행 모드: train 또는 inference")
    p.add_argument("--data_root",   type=str, default="dataset")
    p.add_argument("--pretrained_model", type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--l_face",     type=float, default=1.0)   # λ₁
    p.add_argument("--l_text",     type=float, default=1.0)   # λ₂
    p.add_argument("--l_kl",       type=float, default=0.1)   # λ₃, KL divergence weight
    p.add_argument("--save_dir",   type=str, default="lora_out")
    
    # Inference 관련 옵션들
    p.add_argument("--lora_path", type=str, default=None,
                   help="inference시 로드할 LoRA 가중치 경로")
    p.add_argument("--output_dir", type=str, default="inference_output",
                   help="inference 결과 저장 경로")
    p.add_argument("--num_inference_steps", type=int, default=50,
                   help="inference시 diffusion steps 수")
    p.add_argument("--guidance_scale", type=float, default=7.5,
                   help="classifier-free guidance scale")
    p.add_argument("--strength", type=float, default=0.8,
                   help="img2img strength (0.0-1.0)")
    
    # 데이터셋 관련 옵션들
    p.add_argument("--celeba_dir", type=str,
               default="dataset/celeba-hq/celeba-256",
               help="CelebA-HQ 256px 얼굴 이미지 폴더")
    p.add_argument("--mpii_dir", type=str, 
               default="dataset/mpii-one-person",
               help="MPII 데이터셋 루트 디렉토리")
    
    return p.parse_args()

# ----------------------------
# 2) 데이터셋
# ----------------------------

class MixedFacePoseDataset(Dataset):
    """
    CelebA 얼굴 ↔ MPII 사람 이미지·캡션을 매 스텝 랜덤 매칭
    반환: ref(얼굴), orig(사람+배경), prompt(캡션)
    """
    def __init__(self, celeba_dir, mpii_dir, size):
        self.size = size

        # 1) 얼굴 파일 리스트
        self.face_paths = glob.glob(os.path.join(celeba_dir, "*.jpg"))
        assert len(self.face_paths) > 0, f"No jpg in {celeba_dir}"

        # 2) MPII 이미지+캡션 로드 (로컬 JSON)
        print("Loading MPII dataset...")
        annotations_file = os.path.join(mpii_dir, "mpii_annotations.json")
        images_dir = os.path.join(mpii_dir, "mpii_images")
        
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.mpii_data = []
        for annotation in annotations["image_annotations"]:
            if "image" in annotation:  # 이미지 필드가 있는 항목만 처리
                image_path = os.path.join(images_dir, annotation["image"])
                if os.path.exists(image_path):
                    self.mpii_data.append({
                        "image_path": image_path,
                        "description": annotation["description"]
                    })
        
        assert len(self.mpii_data) > 0, f"No valid MPII images found in {images_dir}"
        print(f"Loaded {len(self.mpii_data)} MPII samples")

        # 3) 공통 transform
        self.tf = transforms.Compose([
            transforms.Resize(size, Image.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])   # [-1,1]
        ])

    def __len__(self):
        return len(self.mpii_data)

    def __getitem__(self, idx):
        # (a) 사람 이미지 & 텍스트
        sample = self.mpii_data[idx]
        image_path = sample["image_path"]
        prompt = sample["description"]
        
        # 이미지 로드
        orig_pil = Image.open(image_path).convert("RGB")

        # (b) 같은 배치 안에서 얼굴은 **아무 이미지 하나 랜덤** 선택
        face_path = random.choice(self.face_paths)
        ref_pil = Image.open(face_path).convert("RGB")

        return {
            "ref": self.tf(ref_pil),
            "orig": self.tf(orig_pil),
            "prompt": prompt
        }
        
class FaceTextDataset(Dataset):
    def __init__(self, root, size):
        self.samples = []
        self.size = size
        for pid in sorted(os.listdir(root)):
            folder = os.path.join(root, pid)
            self.samples.append(
                dict(
                    ref=os.path.join(folder, "ref_face.jpg"),
                    orig=os.path.join(folder, "orig.jpg"),
                    prompt=open(os.path.join(folder, "prompt.txt")).read().strip()
                )
            )
        self.img_tf = transforms.Compose([
            transforms.Resize(size, Image.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1,1] 범위
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ref"   : self.img_tf(Image.open(s["ref"]).convert("RGB")),
            "orig"  : self.img_tf(Image.open(s["orig"]).convert("RGB")),
            "prompt": s["prompt"]
        }

class InferenceDataset(Dataset):
    """
    Inference용 데이터셋 
    각 샘플은 참조 얼굴, 타겟 이미지, 프롬프트로 구성
    """
    def __init__(self, root, size):
        self.samples = []
        self.size = size
        
        for sample_id in sorted(os.listdir(root)):
            folder = os.path.join(root, sample_id)
            if os.path.isdir(folder):
                sample = {
                    "ref_face": os.path.join(folder, "ref_face.jpg"),
                    "target_img": os.path.join(folder, "target_img.jpg"), 
                    "prompt": open(os.path.join(folder, "prompt.txt")).read().strip(),
                    "sample_id": sample_id
                }
                # 모든 필수 파일이 존재하는지 확인
                if all(os.path.exists(sample[key]) for key in ["ref_face", "target_img"]):
                    self.samples.append(sample)
        
        self.img_tf = transforms.Compose([
            transforms.Resize(size, Image.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1,1] 범위
        ])

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ref_face": self.img_tf(Image.open(s["ref_face"]).convert("RGB")),
            "target_img": self.img_tf(Image.open(s["target_img"]).convert("RGB")),
            "prompt": s["prompt"],
            "sample_id": s["sample_id"]
        }

# ----------------------------
# Face Cropping 유틸리티
# ----------------------------
def crop_face_batch(images, mtcnn_detector, target_size=160):
    """
    배치 이미지에서 얼굴을 detection하고 crop합니다.
    Args:
        images: torch.Tensor (B, C, H, W), 값 범위 [0, 1]
        mtcnn_detector: MTCNN 모델
        target_size: crop된 얼굴의 크기
    Returns:
        cropped_faces: torch.Tensor (B, C, target_size, target_size)
    """
    batch_size = images.shape[0]
    device = images.device
    cropped_faces = []
    
    with torch.no_grad():  # gradient flow 차단
        for i in range(batch_size):
            # tensor를 PIL Image로 변환 (0-255 범위)
            img_pil = transforms.ToPILImage()(images[i].cpu())
            
            # 얼굴 detection
            boxes, _ = mtcnn_detector.detect(img_pil)
            
            if boxes is not None and len(boxes) > 0:
                # 첫 번째(가장 큰) 얼굴 박스 사용
                box = boxes[0]
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # 얼굴 영역 crop
                face_crop = img_pil.crop((x1, y1, x2, y2))
                face_crop = face_crop.resize((target_size, target_size), Image.BILINEAR)
                
                # tensor로 변환하여 [0, 1] 범위로 정규화
                face_tensor = transforms.ToTensor()(face_crop)
            else:
                # 얼굴이 detection되지 않으면 center crop 사용
                center_crop = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(min(img_pil.size)),
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor()
                ])
                face_tensor = center_crop(images[i].cpu())
            
            cropped_faces.append(face_tensor)
    
    return torch.stack(cropped_faces).to(device)

# ----------------------------
# 4) Inference 함수
# ----------------------------
def run_inference(args):
    """
    Inference 모드: 학습된 LoRA 가중치를 사용하여 이미지 생성
    """
    print("🚀 Starting inference mode...")
    
    if args.lora_path is None:
        raise ValueError("--lora_path must be specified for inference mode")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Stable Diffusion 파이프라인 초기화
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model, 
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.enable_vae_tiling()
    
    # UNet은 기본 4채널 유지 (batch concatenation 사용)
    print("🔧 Using batch concatenation for reference image integration...")
    
    # LoRA 가중치 로드
    print(f"📦 Loading LoRA weights from {args.lora_path}")
    pipe.unet.load_adapter(args.lora_path)
    pipe = pipe.to("cuda")
    
    # Inference 데이터셋 로드
    print(f"📁 Loading inference dataset from {args.data_root}")
    dataset = InferenceDataset(args.data_root, args.resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"🎯 Found {len(dataset)} samples for inference")
    
    # 커스텀 inference 루프 (reference image 통합)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ref_face = batch["ref_face"][0]      # (C, H, W)
            target_img = batch["target_img"][0]  # (C, H, W)
            prompt = batch["prompt"][0]
            sample_id = batch["sample_id"][0]
            
            print(f"🖼️  Processing sample {i+1}/{len(dataset)}: {sample_id}")
            print(f"    Prompt: {prompt}")
            
            # 배치 차원 추가
            ref_batch = ref_face.unsqueeze(0).to("cuda")    # (1, C, H, W)
            target_batch = target_img.unsqueeze(0).to("cuda")  # (1, C, H, W)
            
            # latent space로 인코딩
            target_latents = pipe.vae.encode(target_batch.half()).latent_dist.sample()
            target_latents = target_latents * pipe.vae.config.scaling_factor
            
            ref_latents = pipe.vae.encode(ref_batch.half()).latent_dist.sample()
            ref_latents = ref_latents * pipe.vae.config.scaling_factor
            
            # text embeddings
            text_inputs = pipe.tokenizer(
                [prompt], padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to("cuda")
            text_embeddings = pipe.text_encoder(**text_inputs)[0]
            
            # Reference image를 CLIP으로 인코딩하여 text embedding에 추가
            ref_clip_inputs = clip_processor(
                images=[transforms.ToPILImage()(ref_face * 0.5 + 0.5)],
                return_tensors="pt",
                padding=True
            ).to("cuda")
            
            ref_img_features = clip_model.get_image_features(ref_clip_inputs['pixel_values'])
            ref_img_features = F.normalize(ref_img_features, dim=-1)
            ref_img_features_expanded = ref_img_features.unsqueeze(1)  # (1, 1, 768)
            enhanced_text_embeddings = torch.cat([text_embeddings, ref_img_features_expanded], dim=1)  # (1, 78, 768)
            
            # unconditioned embeddings for classifier-free guidance
            uncond_inputs = pipe.tokenizer(
                [""], padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to("cuda")
            uncond_embeddings = pipe.text_encoder(**uncond_inputs)[0]
            
            # uncond도 reference features 추가 (빈 reference로)
            empty_ref_features = torch.zeros_like(ref_img_features_expanded)
            enhanced_uncond_embeddings = torch.cat([uncond_embeddings, empty_ref_features], dim=1)
            
            # guidance를 위해 concatenate
            text_embeddings = torch.cat([enhanced_uncond_embeddings, enhanced_text_embeddings])
            
            # noise scheduler 초기화
            pipe.scheduler.set_timesteps(args.num_inference_steps)
            timesteps = pipe.scheduler.timesteps
            
            # 초기 noise로 시작 (img2img이므로 부분적으로 noise)
            noise = torch.randn_like(target_latents)
            init_timestep = min(int(args.num_inference_steps * args.strength), args.num_inference_steps)
            t_start = max(args.num_inference_steps - init_timestep, 0)
            
            # target latent에 noise 추가
            latents = pipe.scheduler.add_noise(target_latents, noise, timesteps[t_start:t_start+1])
            
            # denoising loop
            for i, t in enumerate(timesteps[t_start:]):
                # latent 입력 (메모리 효율적 - reference 없이 target만)
                latent_model_input = latents
                
                # classifier-free guidance를 위해 복제
                latent_model_input = torch.cat([latent_model_input] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
                
                # UNet으로 noise prediction (enhanced text embeddings 사용)
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # 다음 step을 위한 latent 업데이트
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # latent을 이미지로 디코딩
            generated_image = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).half()).sample
            generated_image = torch.clamp(generated_image, -1, 1)
            
            # [-1,1] 범위를 [0,1] 범위로 변환 후 PIL Image로 변환
            generated_pil = transforms.ToPILImage()(generated_image[0] * 0.5 + 0.5)
            target_pil = transforms.ToPILImage()(target_img * 0.5 + 0.5)
            ref_face_pil = transforms.ToPILImage()(ref_face * 0.5 + 0.5)
            
            # 결과 저장
            output_path = os.path.join(args.output_dir, f"{sample_id}_generated.jpg")
            generated_pil.save(output_path)
            
            # 비교용 원본 이미지들도 저장
            ref_face_pil.save(os.path.join(args.output_dir, f"{sample_id}_ref_face.jpg"))
            target_pil.save(os.path.join(args.output_dir, f"{sample_id}_target.jpg"))
            
            print(f"    ✅ Saved: {output_path}")
    
    print(f"🎉 Inference completed! Results saved in {args.output_dir}")

# ----------------------------
# 5) 메인 학습 루프
# ----------------------------
def train_main(args):
    accelerator = Accelerator(gradient_accumulation_steps=1)

    # (a) Stable Diffusion img2img 파이프 구성
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16
    )
    pipe.enable_vae_tiling()  # 큰 해상도 대응
    pipe.safety_checker = None   # 개인 연구 용도

    # LoRA 주입 (UNet cross-attention 만)
    unet = pipe.unet
    
    # UNet은 기본 4채널 유지 (batch concatenation 사용)
    print("🔧 Using batch concatenation for reference image integration...")
    
    # KL Loss 계산을 위해 원본 UNet 복사
    unet_original = copy.deepcopy(unet) 
    unet_original = unet_original.to(accelerator.device) # 원본 UNet을 올바른 디바이스로 이동
    for p in unet_original.parameters(): p.requires_grad_(False) # 원본 UNet은 freeze
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none", modules_to_save=None
    )
    unet_lora = get_peft_model(unet, lora_config)
    unet_lora.print_trainable_parameters()

    # (b) CLIP - 텍스트/이미지
    clip_model      = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    for p in clip_model.parameters(): p.requires_grad_(False)

    # (c) CLIP-ReID / 얼굴 임베딩
    # torchreid 모델 초기화 (OSNet)
    reid_model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,  # 임시 클래스 수 (feature extraction만 사용)
        loss="softmax",
        pretrained=True
    )
    reid_model = reid_model.to(accelerator.device)
    reid_model.eval()  # feature extraction 모드
    
    # reid 모델 파라미터를 frozen 상태로 설정
    for param in reid_model.parameters():
        param.requires_grad = False
    
    # feature extraction 함수 정의
    def extract_reid_features(images):
        """
        torchreid 모델을 사용한 feature extraction
        Args:
            images: torch.Tensor (B, C, H, W), 값 범위 [-1, 1]
        Returns:
            features: torch.Tensor (B, feature_dim)
        """
        with torch.no_grad():
            # [-1,1] 범위를 [0,1] 범위로 변환
            images_norm = images * 0.5 + 0.5
            
            # torchreid는 ImageNet 정규화를 사용
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            # 이미지 크기 조정 (torchreid 기본: 256x128)
            resize_transform = transforms.Compose([
                transforms.Resize((256, 128), antialias=True),
                normalize
            ])
            
            batch_size = images.shape[0]
            features = []
            
            for i in range(batch_size):
                img = resize_transform(images_norm[i])
                img = img.unsqueeze(0).to(images.device)  # (1, C, H, W)
                
                # feature extraction
                feat = reid_model(img)
                if isinstance(feat, tuple):
                    feat = feat[0]  # 첫 번째 output만 사용
                features.append(feat.squeeze(0))
            
            return torch.stack(features)

    # (d) MTCNN face detector 초기화 (frozen)
    mtcnn_detector = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
        device=accelerator.device, keep_all=False
    )
    # MTCNN 파라미터들을 frozen 상태로 설정
    for param in mtcnn_detector.parameters():
        param.requires_grad = False

    # (e) 옵티마이저 (LoRA 파라미터만)
    optimizer = torch.optim.AdamW(unet_lora.parameters(), lr=args.lr)

    # (f) 데이터
    dataset  = MixedFacePoseDataset(args.celeba_dir, args.mpii_dir, args.resolution)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # accelerator.prepare로 모든 컴포넌트를 준비
    unet_lora, clip_model, optimizer, loader = accelerator.prepare(
        unet_lora, clip_model, optimizer, loader
    )
    
    # VAE, text_encoder, tokenizer도 같은 디바이스로 이동
    pipe.vae = pipe.vae.to(accelerator.device)
    pipe.text_encoder = pipe.text_encoder.to(accelerator.device)
    
    # 디바이스 확인 로그
    print(f"🖥️  Device info:")
    print(f"   Accelerator device: {accelerator.device}")
    print(f"   UNet device: {next(unet_lora.parameters()).device}")
    print(f"   VAE device: {next(pipe.vae.parameters()).device}")
    print(f"   CLIP device: {next(clip_model.parameters()).device}")
    print(f"   ReID device: {next(reid_model.parameters()).device}")

    # 스케줄러 – SD 훈련과 동일(DDPM)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    # 로그 파일 설정
    if accelerator.is_main_process:
        log_dir = os.path.join(args.save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # CSV 헤더 작성
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'epoch', 'global_step', 'loss_face', 'loss_text', 'loss_kl', 'total_loss'])
        
        print(f"📝 Training log will be saved to: {log_file}")

    global_step = 0
    for epoch in tqdm(range(args.epochs), desc="Epochs", position=0):
        unet_lora.train()
        
        # Create progress bar for batches
        batch_pbar = tqdm(loader, 
                         desc=f"Epoch {epoch+1}/{args.epochs}", 
                         position=1, 
                         leave=False)
        
        for batch in batch_pbar:
            with accelerator.accumulate(unet_lora):
                orig = batch["orig"]      # (B,3,H,W)   [-1,1]
                ref  = batch["ref"]       # (B,3,H,W)
                prompts = batch["prompt"]
                
                # 데이터를 올바른 디바이스로 이동
                orig = orig.to(accelerator.device)
                ref = ref.to(accelerator.device)

                # ---------- ➊ SD forward/backward ----------
                # encode 원본 → latent (VAE는 float16이므로 입력도 맞춰줌)
                latents = pipe.vae.encode(orig.half()).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                
                # encode reference image → latent
                ref_latents = pipe.vae.encode(ref.half()).latent_dist.sample()
                ref_latents = ref_latents * pipe.vae.config.scaling_factor

                bsz = latents.shape[0]
                t = torch.randint(0, noise_scheduler.num_train_timesteps,
                                  (bsz,), device=latents.device).long()
                noise = torch.randn_like(latents)
                ref_noise = torch.randn_like(ref_latents)
                
                # 둘 다 같은 timestep으로 noise 추가
                noisy_latents = noise_scheduler.add_noise(latents, noise, t)
                noisy_ref_latents = noise_scheduler.add_noise(ref_latents, ref_noise, t)
                
                # Reference image를 CLIP으로 인코딩하여 text embedding에 추가
                # 먼저 reference image를 PIL로 변환
                ref_images_pil = []
                for ref_img in ref:
                    ref_pil = transforms.ToPILImage()(ref_img * 0.5 + 0.5)
                    ref_images_pil.append(ref_pil)
                
                # CLIP으로 reference image 인코딩
                ref_clip_inputs = clip_processor(
                    images=ref_images_pil,
                    return_tensors="pt",
                    padding=True
                ).to(accelerator.device)
                
                ref_img_features = clip_model.get_image_features(ref_clip_inputs['pixel_values'])
                ref_img_features = F.normalize(ref_img_features, dim=-1)
                
                # text embeddings
                text_inputs = pipe.tokenizer(
                    list(prompts), 
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(latents.device)
                text_embeddings = pipe.text_encoder(**text_inputs)[0]
                
                # Reference image features를 text embedding에 concatenate
                # text_embeddings: (B, 77, 768), ref_img_features: (B, 768)
                ref_img_features_expanded = ref_img_features.unsqueeze(1)  # (B, 1, 768)
                enhanced_text_embeddings = torch.cat([text_embeddings, ref_img_features_expanded], dim=1)  # (B, 78, 768)
                
                # 데이터 타입을 UNet과 맞춤 (float16)
                enhanced_text_embeddings = enhanced_text_embeddings.half()

                # UNet forward with enhanced text embeddings (메모리 2배 증가 없음!)
                noise_pred = unet_lora(
                    noisy_latents,  # 원본 latents만 사용
                    t,
                    encoder_hidden_states=enhanced_text_embeddings
                ).sample

                # ---------- ➋ 완전한 denoising으로 x0 생성 ----------
                # 전체 denoising process를 통해 최종 이미지 생성
                with torch.no_grad():
                    # noise scheduler로 전체 denoising 수행
                    scheduler_temp = DDPMScheduler(
                        num_train_timesteps=1000, 
                        beta_schedule="squaredcos_cap_v2"
                    )
                    scheduler_temp.set_timesteps(50)  # inference steps
                    
                    # scheduler의 timesteps를 올바른 디바이스로 이동
                    scheduler_temp.timesteps = scheduler_temp.timesteps.to(t.device)
                    
                    # 현재 timestep에서 시작
                    current_latents = noisy_latents.clone()
                    
                    # 현재 timestep부터 0까지 denoising
                    timesteps = scheduler_temp.timesteps.to(t.device)
                    current_t_idx = torch.where(timesteps <= t[0])[0]
                    if len(current_t_idx) > 0:
                        start_idx = current_t_idx[0].item()
                    else:
                        start_idx = 0
                    
                    for step_t in timesteps[start_idx:]:
                        # UNet prediction with enhanced text embeddings
                        step_noise_pred = unet_lora(
                            current_latents,
                            step_t.unsqueeze(0).repeat(bsz).to(current_latents.device),
                            encoder_hidden_states=enhanced_text_embeddings
                        ).sample
                        
                        # scheduler step - 원본 이미지만 업데이트
                        current_latents = scheduler_temp.step(
                            step_noise_pred, step_t.cpu(), current_latents
                        ).prev_sample
                    
                    # 최종 denoised latents를 이미지로 변환
                    imgs_gen = pipe.vae.decode(
                        (current_latents / pipe.vae.config.scaling_factor).half()).sample
                imgs_gen = torch.clamp(imgs_gen, -1, 1)

                # ---------- ➌ CLIP-ReID 얼굴 손실 ----------
                # 얼굴 detection & crop 적용
                # 먼저 [-1,1] 범위를 [0,1] 범위로 변환
                ref_imgs_norm = ref * 0.5 + 0.5  # [-1,1] → [0,1]
                gen_imgs_norm = imgs_gen * 0.5 + 0.5  # [-1,1] → [0,1]
                
                # 얼굴 영역 crop (gradient flow 차단)
                with torch.no_grad():
                    ref_faces = crop_face_batch(ref_imgs_norm, mtcnn_detector, target_size=224)
                    gen_faces = crop_face_batch(gen_imgs_norm, mtcnn_detector, target_size=224)
                
                # ReID feature extraction을 위해 [0,1] → [-1,1] 범위로 변환
                ref_faces_reid = ref_faces * 2.0 - 1.0  # [0,1] → [-1,1]
                gen_faces_reid = gen_faces * 2.0 - 1.0  # [0,1] → [-1,1]
                
                # CLIP-ReID 특징 추출 (crop된 얼굴 사용)
                ref_feats = extract_reid_features(ref_faces_reid)   # (B, 512)
                gen_feats = extract_reid_features(gen_faces_reid)   # (B, 512)
                sim_face  = F.cosine_similarity(ref_feats, gen_feats, dim=1)
                loss_face = (1 - sim_face).mean()      # 높을수록 좋으니 1-cosine

                # ---------- ➍ CLIP 텍스트 손실 ----------
                # 생성된 이미지를 PIL Image 형태로 변환 (CLIP processor용)
                gen_images_pil = []
                for g in imgs_gen:
                    img_pil = transforms.ToPILImage()(g * 0.5 + 0.5)
                    gen_images_pil.append(img_pil)
                
                clip_inputs = clip_processor(
                    text=list(prompts),
                    images=gen_images_pil,
                    return_tensors="pt",
                    padding=True
                )
                
                # CLIP 입력을 올바른 디바이스로 이동
                clip_inputs = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in clip_inputs.items()}

                img_emb = clip_model.get_image_features(clip_inputs['pixel_values'])
                txt_emb = clip_model.get_text_features(clip_inputs['input_ids'],
                                                       attention_mask=clip_inputs.get('attention_mask', None))
                img_emb, txt_emb = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
                sim_text = (img_emb * txt_emb).sum(-1)
                loss_text = (1 - sim_text).mean()

                # ---------- ➎ 총 손실 ----------
                # KL divergence loss 계산 (안정적인 버전)
                with torch.no_grad():
                    original_output = unet_original(
                        noisy_latents,
                        t,
                        encoder_hidden_states=enhanced_text_embeddings
                    ).sample
                
                # 안정적인 KL divergence 계산
                # 1) 값들을 clamp해서 극값 방지
                noise_pred_clamped = torch.clamp(noise_pred, -10, 10)
                original_output_clamped = torch.clamp(original_output, -10, 10)
                
                # 2) temperature scaling으로 분포를 부드럽게 만듦
                temperature = 2.0
                log_p = F.log_softmax(noise_pred_clamped / temperature, dim=1)
                q = F.softmax(original_output_clamped / temperature, dim=1)
                
                # 3) KL divergence 계산 (안정적인 버전)
                kl_div = F.kl_div(log_p, q, reduction='batchmean')
                
                # 4) NaN 체크 및 처리
                if torch.isnan(kl_div) or torch.isinf(kl_div):
                    # print(f"⚠️  KL divergence is {kl_div.item()}, setting to 0")
                    kl_div = torch.tensor(0.0, device=kl_div.device, requires_grad=True)
                
                # 총 손실 계산
                loss = args.l_face * loss_face + args.l_text * loss_text + args.l_kl * kl_div
                accelerator.backward(loss)
                optimizer.step(); optimizer.zero_grad()

                global_step += 1
                
                # Update progress bar with current loss values
                if global_step % 10 == 0:
                    batch_pbar.set_postfix({
                        'L_face': f'{loss_face.item():.3f}',
                        'L_text': f'{loss_text.item():.3f}',
                        'L_kl': f'{kl_div.item():.3f}',
                        'total': f'{loss.item():.3f}'
                    })
                
                # 1000스텝마다 로그 파일에 기록
                if accelerator.is_main_process and global_step % 1000 == 0:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            epoch + 1,
                            global_step,
                            f"{loss_face.item():.6f}",
                            f"{loss_text.item():.6f}",
                            f"{kl_div.item():.6f}",
                            f"{loss.item():.6f}"
                        ])
                    print(f"📝 Logged training metrics at step {global_step} to {log_file}")

        # epoch-단위 LoRA 가중치 저장
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            unet_lora.save_pretrained(os.path.join(args.save_dir, f"epoch{epoch:02d}"))
            print(f"✅ Epoch {epoch+1} completed. LoRA weights saved to {args.save_dir}/epoch{epoch:02d}")

def main():
    args = parse_args()
    
    if args.mode == "train":
        print("🏋️ Starting training mode...")
        train_main(args)
    elif args.mode == "inference":
        print("🎯 Starting inference mode...")
        run_inference(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
