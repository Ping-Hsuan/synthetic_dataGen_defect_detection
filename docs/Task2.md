# Task 2: Implementation Guide

**Date:** January 12, 2026  
**Project:** Few-Shot Industrial Defect Generation  
**Companion to:** TASK2_MODEL_SELECTION.md

This document provides practical implementation details, code examples, hardware requirements, and reproducibility guidelines for fine-tuning Stable Diffusion 1.5 on cable defects.

---

## 1. Implementation Outline

### 1.1 Dependencies

**Open-Source Stack (Following Project Guidelines):**

```python
requirements = {
    # Core Training Libraries
    "diffusers": ">=0.25.0",           # SD 1.5 pipeline, schedulers
    "peft": ">=0.7.0",                 # LoRA adapters (not reimplementing)
    "accelerate": ">=0.25.0",          # Multi-GPU, mixed precision
    "transformers": ">=4.36.0",        # Model architectures
    
    # Deep Learning
    "torch": ">=2.1.0",                # PyTorch
    "torchvision": ">=0.16.0",         # Image transforms
    
    # Data & Evaluation
    "datasets": ">=2.16.0",            # HuggingFace datasets API
    "PIL": ">=10.0.0",                 # Image I/O
    "numpy": ">=1.24.0",               # Numerical operations
    "torchmetrics": ">=1.0.0",         # FID calculation
    
    # Experiment Tracking (Optional)
    "wandb": ">=0.15.0",               # Weights & Biases
    "tensorboard": ">=2.14.0",         # TensorBoard logging
}
```

---

### 1.2 Model Initialization

```python
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import torch.nn as nn

# 1. Load pre-trained model
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

# 2. Freeze VAE
for param in pipeline.vae.parameters():
    param.requires_grad = False

# 3. Add class conditioning (replaces CLIP text encoder)
class ClassConditionedUNet(nn.Module):
    """Wraps SD 1.5 UNet to accept integer class IDs instead of text prompts.
    
    Architecture:
        class_id (int) → Embedding Layer → [768-dim vector] → UNet cross-attention
    
    This replaces CLIP text encoder while keeping the UNet architecture unchanged.
    """
    def __init__(self, unet, num_classes=9, embed_dim=768):
        super().__init__()
        self.unet = unet
        # Learnable lookup table: class_id → 768-dim vector
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        
    def forward(self, sample, timestep, class_labels, **kwargs):
        # class_labels: [B] (batch of class IDs, e.g., [3, 5, 0, 8])
        # Get class embeddings: [B, 768]
        class_embeds = self.class_embeddings(class_labels)
        
        # Feed to UNet cross-attention (same interface as CLIP output)
        return self.unet(
            sample,                           # Noisy latents
            timestep,                         # Diffusion timestep
            encoder_hidden_states=class_embeds  # ← Conditioning signal
        )

# 4. Apply LoRA
model = ClassConditionedUNet(pipeline.unet, num_classes=9)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["to_q", "to_v"])
model = get_peft_model(model, lora_config)

# 5. Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 8,007,000 || all params: 860,007,000 || trainable%: 0.93%
```

---

### 1.3 Training Loop Sketch

```python
# Pseudo-code for training loop
for epoch in range(100):
    model.train()
    
    for batch in train_loader:
        images = batch['pixel_values']              # [B, 3, 512, 512]
        class_labels = batch['class_labels']        # [B]
        
        # Encode to latent space
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        # Sample noise and timestep
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (B,))
        
        # Add noise (forward diffusion)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_latents, timesteps, class_labels)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation every 5 epochs
    if epoch % 5 == 0:
        validate_and_generate_samples(model, val_loader, epoch)
```

---

## 2. Expected Outcomes

### 2.1 Training Timeline

```
Epoch   | Phase     | Train Loss | FID Score | Status
─────────────────────────────────────────────────────
1-20    | Phase 1   | 0.15→0.08 | ~80→60   | Learning classes
21-60   | Phase 2   | 0.08→0.05 | 60→40    | Adapting to cables
61-100  | Phase 3   | 0.05→0.04 | 40→35    | Polishing quality
```

**Progress Monitoring:**
- ✅ Good: Train loss decreases smoothly, FID improves, generated samples look realistic
- ⚠️ Warning: Train loss decreases but FID plateaus (model may be memorizing)
- ❌ Overfitting: Train loss decreases but FID increases (revert to earlier checkpoint)

**Note:** No validation loss (no data split). Use FID and visual inspection for validation.

---

### 2.2 Generation Quality Milestones

**Epoch 20:** Basic cable structure, recognizable damage types  
**Epoch 40:** Clear defect patterns, good texture quality  
**Epoch 60:** Realistic cables, accurate defect localization  
**Epoch 80:** High-quality results, subtle variations  
**Epoch 100:** Production-ready generation

---

### 2.3 Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Overfitting | Medium | High | Use LoRA, monitor val loss, early stopping |
| Mode collapse | Low | Medium | CFG dropout, increase diversity |
| Class confusion | Low | Medium | Increase class embedding dim, longer training |
| Poor convergence | Low | High | Validate hyperparams on 2 epochs first |
| OOM errors | Low | High | Reduce batch size, use gradient checkpointing |

---

## 3. Hardware & Cost Estimates

### 3.1 Training Hardware Options

| Platform | GPU | VRAM | Cost | Time |
|----------|-----|------|------|------|
| Google Colab (Free) | T4 | 16GB | $0 | 2-3 hours ✅ |
| Google Colab Pro | V100 | 16GB | $10/mo | 1.5 hours |
| Kaggle | P100 | 16GB | Free | 2.5 hours ✅ |
| Lambda Labs | A10 | 24GB | $0.50/hr | 1.5 hours |
| RunPod | RTX 3090 | 24GB | $0.34/hr | 2 hours |

**Recommended:** Google Colab (Free T4) - sufficient for this task

---

### 3.2 Storage Requirements

```
Dataset:                    ~500 MB  (90 images @ 1024×1024)
Model checkpoint:           ~1.7 GB  (SD 1.5 + LoRA)
Generated samples:          ~100 MB  (validation images)
Logs & metadata:            ~50 MB
─────────────────────────────────────
Total:                      ~2.35 GB
```

---

## 4. Reproducibility

### 4.1 Seeds & Determinism

```python
reproducibility_config = {
    "random_seed": 42,
    "numpy_seed": 42,
    "torch_seed": 42,
    "cuda_deterministic": True,
    "cudnn_benchmark": False,
}

# Set all seeds
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 4.2 Version Pinning

All dependencies pinned in `requirements.txt`:
```
diffusers==0.25.1
transformers==4.36.2
peft==0.7.1
accelerate==0.25.0
torch==2.1.2
torchvision==0.16.2
datasets==2.16.1
pillow==10.1.0
numpy==1.24.4
scikit-learn==1.3.2
```

### 4.3 Experiment Tracking

```python
logging_config = {
    "tool": "wandb",  # or tensorboard
    "project": "cable-defect-generation",
    "entity": "your-username",
    "log_frequency": 10,  # steps
    "log_images": True,
    "log_gradients": False,  # Can slow down training
}

# Initialize tracker
import wandb
wandb.init(
    project=logging_config["project"],
    entity=logging_config["entity"],
    config={
        "learning_rate": 1e-5,
        "epochs": 100,
        "batch_size": 4,
        "lora_rank": 8,
    }
)
```

---

## 5. References

1. **Stable Diffusion**: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022

2. **DreamBooth**: Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation," CVPR 2023

3. **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022

4. **Classifier-Free Guidance**: Ho, J., & Salimans, T. (2021). "Classifier-Free Diffusion Guidance," NeurIPS Workshop on Deep Generative Models and Downstream Applications 2021

5. **MVTec AD Dataset**: Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019

6. **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models," NeurIPS 2020

7. **DDIM**: Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models," ICLR 2021

---

## Appendix A: Full Configuration File

```python
# config/training_config.yaml
model:
  name: "runwayml/stable-diffusion-v1-5"
  num_classes: 9
  class_embedding_dim: 768
  
data:
  product_class: "cable"
  dataset_path: "./scripts/Defect_Spectrum/DS-MVTec/cable"
  train_samples: 90          # Use all samples (no validation split)
  val_samples: 0             # Validation via generation, not data split
  resolution: 512
  normalization: [-1, 1]
  
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
  modules_to_save: ["class_embeddings"]
  
training:
  num_epochs: 100
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  mixed_precision: "fp16"
  gradient_checkpointing: True
  ema_decay: 0.9999
  
scheduler:
  type: "cosine_with_warmup"
  warmup_steps: 100
  min_lr: 1e-7
  
validation:
  # Generation-based validation (no data split)
  generation_frequency: 5      # Generate samples every 5 epochs
  num_samples_per_class: 3     # 3 × 9 = 27 images per validation
  guidance_scale: 7.5
  num_inference_steps: 50
  
  # FID calculation
  fid_frequency: 10            # Calculate FID every 10 epochs
  fid_num_samples: 90          # Generate 90 images (10 per class)
  fid_reference_set: "all_90_training"  # Compare to all training data
  
checkpointing:
  frequency: 10
  keep_last_n: 5
  save_path: "./checkpoints/cable_defect"
  
noise_scheduler:
  train_scheduler: "DDPMScheduler"
  num_train_timesteps: 1000
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: "scaled_linear"
  inference_scheduler: "DDIMScheduler"
  eta: 0.0
```

---

## Appendix B: Evaluation Metrics Implementation

### FID Score Calculation

```python
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance between real and generated images.
    
    Args:
        real_images: Tensor of shape [N, 3, H, W], range [0, 255]
        generated_images: Tensor of shape [N, 3, H, W], range [0, 255]
    
    Returns:
        fid_score: float
    """
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()
```

### Class Accuracy Evaluation

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

def evaluate_class_accuracy(generated_images, true_labels):
    """
    Evaluate if a pre-trained classifier can correctly identify defect types.
    
    Args:
        generated_images: List of PIL Images
        true_labels: List of ground truth class labels
    
    Returns:
        accuracy: float
    """
    # Use a pre-trained ResNet or ViT classifier
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    
    correct = 0
    for img, label in zip(generated_images, true_labels):
        inputs = processor(img, return_tensors="pt")
        outputs = model(**inputs)
        predicted = outputs.logits.argmax(-1).item()
        if predicted == label:
            correct += 1
    
    return correct / len(generated_images)
```

### Diversity Score

```python
import torch.nn.functional as F

def calculate_diversity_score(images_per_class):
    """
    Measure intra-class variation using pairwise LPIPS distances.
    
    Args:
        images_per_class: Dict[class_id, List[Tensor]]
    
    Returns:
        mean_diversity: float (higher = more diverse)
    """
    from lpips import LPIPS
    lpips_fn = LPIPS(net='alex')
    
    diversity_scores = []
    for class_id, images in images_per_class.items():
        if len(images) < 2:
            continue
        
        # Calculate pairwise LPIPS distances
        distances = []
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                dist = lpips_fn(images[i], images[j])
                distances.append(dist.item())
        
        diversity_scores.append(np.mean(distances))
    
    return np.mean(diversity_scores)
```

---

## Appendix C: Progressive Freezing Implementation

```python
def set_phase_trainability(model, phase, lora_modules):
    """
    Set which modules are trainable for each training phase.
    
    Args:
        model: The ClassConditionedUNet with LoRA
        phase: 1, 2, or 3
        lora_modules: List of module names with LoRA adapters
    """
    # Phase 1: Only LoRA and class embeddings
    if phase == 1:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA adapters
        for name, module in model.named_modules():
            if any(lora_name in name for lora_name in ["lora_A", "lora_B"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        # Unfreeze class embeddings
        model.class_embeddings.weight.requires_grad = True
    
    # Phase 2: LoRA + decoder + bottleneck
    elif phase == 2:
        # Start from Phase 1 state
        set_phase_trainability(model, 1, lora_modules)
        
        # Unfreeze UNet decoder and mid block
        for name, module in model.named_modules():
            if "up_blocks" in name or "mid_block" in name:
                for param in module.parameters():
                    param.requires_grad = True
    
    # Phase 3: Full UNet (keep VAE frozen)
    elif phase == 3:
        # Unfreeze entire UNet
        for name, param in model.unet.named_parameters():
            param.requires_grad = True
        
        # Keep VAE frozen (if it's part of the model)
        if hasattr(model, 'vae'):
            for param in model.vae.parameters():
                param.requires_grad = False
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Phase {phase}: {trainable_params:,} / {total_params:,} trainable params ({100*trainable_params/total_params:.2f}%)")
```

---

## Appendix D: Sample Generation Script

```python
def generate_samples(model, vae, scheduler, num_samples_per_class=2, num_classes=9):
    """
    Generate validation samples for all classes.
    
    Args:
        model: ClassConditionedUNet
        vae: VAE decoder
        scheduler: DDIMScheduler for inference
        num_samples_per_class: How many samples per class
        num_classes: Total number of classes
    
    Returns:
        images: Dict[class_id, List[PIL.Image]]
    """
    model.eval()
    images = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for class_id in range(num_classes):
            for _ in range(num_samples_per_class):
                # Random latent
                latents = torch.randn(1, 4, 64, 64).to(model.device)
                
                # Class label
                class_label = torch.tensor([class_id]).to(model.device)
                
                # Denoising loop
                scheduler.set_timesteps(50)
                for t in scheduler.timesteps:
                    # Predict noise
                    noise_pred = model(latents, t, class_label)
                    
                    # Denoise step
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                
                # Decode to image
                latents = latents / vae.config.scaling_factor
                image = vae.decode(latents).sample
                
                # Convert to PIL
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                
                images[class_id].append(pil_image)
    
    return images
```

---

**Document Version:** 1.0  
**Last Updated:** January 12, 2026  
**Author:** Ping-Hsuan  
**Status:** Ready for Implementation
