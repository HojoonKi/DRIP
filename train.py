import os, math, random, argparse, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
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

# ----------------------------
# 1) 하이퍼파라미터 & 인자
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="dataset")
    p.add_argument("--pretrained_model", type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int, default=3)
    p.add_argument("--l_face",     type=float, default=1.0)   # λ₁
    p.add_argument("--l_text",     type=float, default=1.0)   # λ₂
    p.add_argument("--save_dir",   type=str, default="lora_out")
    return p.parse_args()

# ----------------------------
# 2) 데이터셋
# ----------------------------
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
# 3) 메인 학습 루프
# ----------------------------
def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=1)

    # (a) Stable Diffusion img2img 파이프 구성
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16
    )
    pipe.enable_vae_tiling()  # 큰 해상도 대응
    pipe.safety_checker = None   # 개인 연구 용도

    # LoRA 주입 (UNet cross-attention 만)
    unet = pipe.unet
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
            images: torch.Tensor (B, C, H, W), 값 범위 [0, 1]
        Returns:
            features: torch.Tensor (B, feature_dim)
        """
        with torch.no_grad():
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
                img = resize_transform(images[i])
                img = img.unsqueeze(0)  # (1, C, H, W)
                
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
    dataset  = FaceTextDataset(args.data_root, args.resolution)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    unet_lora, clip_model, optimizer, loader = accelerator.prepare(
        unet_lora, clip_model, optimizer, loader
    )

    # 스케줄러 – SD 훈련과 동일(DDPM)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
    )

    global_step = 0
    for epoch in range(args.epochs):
        unet_lora.train()
        for batch in loader:
            with accelerator.accumulate(unet_lora):
                orig = batch["orig"]      # (B,3,H,W)   [-1,1]
                ref  = batch["ref"]       # (B,3,H,W)
                prompts = batch["prompt"]

                # ---------- ➊ SD forward/backward ----------
                # encode 원본 → latent
                latents = pipe.vae.encode(orig).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

                bsz = latents.shape[0]
                t = torch.randint(0, noise_scheduler.num_train_timesteps,
                                  (bsz,), device=latents.device).long()
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, t)

                # text embeddings
                text_inputs = pipe.tokenizer(
                    list(prompts), padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(latents.device)
                text_embeddings = pipe.text_encoder(**text_inputs)[0]

                # UNet forward
                noise_pred = unet_lora(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

                # simple MSE loss between pred & true noise (optional, small weight)
                loss_recon = F.mse_loss(noise_pred.float(), noise.float())

                # ---------- ➋ 이미지 재생성(한 스텝만 반전) ----------
                with torch.no_grad():
                    denoised_latents = noisy_latents - noise_pred
                    imgs_gen = pipe.vae.decode(
                        denoised_latents / pipe.vae.config.scaling_factor).sample
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
                
                # ReID feature extraction을 위해 다시 [-1,1] 범위로 변환
                ref_faces_reid = ref_faces * 2.0 - 1.0  # [0,1] → [-1,1]
                gen_faces_reid = gen_faces * 2.0 - 1.0  # [0,1] → [-1,1]
                
                # CLIP-ReID 특징 추출 (crop된 얼굴 사용)
                ref_feats = extract_reid_features(ref_faces_reid.half())   # (B, 512)
                gen_feats = extract_reid_features(gen_faces_reid.half())   # (B, 512)
                sim_face  = F.cosine_similarity(ref_feats, gen_feats, dim=1)
                loss_face = (1 - sim_face).mean()      # 높을수록 좋으니 1-cosine

                # ---------- ➍ CLIP 텍스트 손실 ----------
                clip_inputs = clip_processor(
                    text=list(prompts),
                    images=[(g * 0.5 + 0.5) for g in imgs_gen],
                    return_tensors="pt",
                    padding=True
                ).to(latents.device)

                img_emb = clip_model.get_image_features(clip_inputs.pixel_values)
                txt_emb = clip_model.get_text_features(clip_inputs.input_ids,
                                                       attention_mask=clip_inputs.attention_mask)
                img_emb, txt_emb = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
                sim_text = (img_emb * txt_emb).sum(-1)
                loss_text = (1 - sim_text).mean()

                # ---------- ➎ 총 손실 ----------
                loss = args.l_face * loss_face + args.l_text * loss_text + 0.1 * loss_recon
                accelerator.backward(loss)
                optimizer.step(); optimizer.zero_grad()

                global_step += 1
                if accelerator.is_main_process and global_step % 50 == 0:
                    print(f"step {global_step:05d} | "
                          f"L_face {loss_face.item():.3f} "
                          f"L_text {loss_text.item():.3f}")

        # epoch-단위 LoRA 가중치 저장
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            unet_lora.save_pretrained(os.path.join(args.save_dir, f"epoch{epoch:02d}"))

if __name__ == "__main__":
    main()
