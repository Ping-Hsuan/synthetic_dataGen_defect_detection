# Task 2: Pre-trained Diffusion Model Selection

**Date:** January 12, 2026  
**Project:** Few-Shot Industrial Defect Generation  
**Dataset:** Cable Defects (DS-MVTec)  
**Training Samples:** 72 images (9 classes × 8 samples)

---

## 1. Model Selection & Rationale

### **Selected Model: Stable Diffusion 1.5**
- **HuggingFace ID:** `runwayml/stable-diffusion-v1-5`
- **Architecture:** Latent Diffusion Model (LDM)
- **Parameters:** 860M total, ~8M trainable with LoRA (rank=8)
- **License:** CreativeML Open RAIL-M

---

## 2. Rationale

### 2.1 Comparison with Alternative Models

#### **A. Stable Diffusion 1.5 vs Stable Diffusion 2.1**

| Criterion | SD 1.5 | SD 2.1 | Advantage |
|-----------|--------|--------|-----------|
| Parameters | 860M | 1.4B | SD 1.5 (63% smaller) |
| Overfitting Risk | Lower | Higher with 72 samples | SD 1.5 |
| Training Time | 2-3 hours | 4-6 hours | SD 1.5 |
| VRAM Required | 4-6GB | 8-10GB | SD 1.5 |
| Few-shot Examples | Extensive | Limited | SD 1.5 |
| Community Support | Mature (3+ years) | Less documented | SD 1.5 |
| Base Image Quality | Very Good | Slightly Better | SD 2.1 |

**Decision:** With LoRA (rank=8), we train only ~8M parameters. This gives us **9 samples per million trainable params** (72 samples / 8M params), which is in the safe zone for few-shot learning. While larger models like SD 2.1 (1.4B) or SDXL (2.6B) have better base quality, even with LoRA they require training more parameters (~13M and ~20M respectively), worsening the data-to-parameter ratio and increasing overfitting risk.

Additionally, SD 1.5's smaller architecture is better-proven for LoRA fine-tuning with extensive community validation over 3+ years.

#### **B. Why Not Flux, SDXL, or Larger Models?**

Even with LoRA, larger models require proportionally more trainable parameters:
- **SDXL (2.6B)**: ~20M trainable params → 3.6 samples per million (too low)
- **Flux.1 Dev (12B)**: ~100M+ trainable params → <1 sample per million (catastrophic)

With only 72 samples, these ratios lead to severe overfitting despite LoRA's regularization.

#### **C. Why Not Proprietary Models?**

DALL-E 3, Midjourney, Imagen don't allow fine-tuning on custom data, which is required for Task 4.

---

### 2.2 Key Advantages of SD 1.5 for This Task

1. **Parameter Efficiency with LoRA:** Only 8M trainable parameters (0.93% of model) for 72 samples, giving 9 samples per million trainable params

2. **Proven Few-Shot Track Record:** DreamBooth paper used SD 1.4/1.5; 100,000+ community LoRA models trained on <100 images; extensive industrial use case documentation

3. **Training Efficiency:** 2-3 hours on free T4 GPU (Google Colab), vs 6-12 hours for larger models

4. **Computational Accessibility:** Fits on free cloud resources with LoRA fine-tuning

---

## 3. Data Configuration

### Dataset Specification

```python
dataset_config = {
    "product_class": "cable",
    "dataset_source": "DS-MVTec",
    "damage_types": 9,  # Including "good"
    "total_samples": 90,
    "samples_per_damage_type": 10,
    
    # Training Configuration
    "train_samples": 90,       # Use ALL samples (no holdout)
    "val_samples": 0,          # No validation split
    
    # Image Properties
    "source_resolution": (1024, 1024),
    "target_resolution": 512,  # Downsample for SD 1.5
    "format": "RGB",
}
```

**Justification:**
- **No validation split:** With only 90 samples, validation loss from 18 samples would be too noisy. Instead, use generation-based validation (see Section 5.3).
- **Use all available data:** LoRA's strong regularization (only 8M trainable params) prevents overfitting, allowing safe use of all 90 samples.
- **512×512 resolution** matches SD 1.5 native training resolution, reduces compute by 4×.
- **Minimal augmentation** (horizontal flip only) preserves defect appearance.

---

## 4. Open-Source Tools

**Following Project Guidelines:** Reusing existing open-source tools rather than reimplementing from scratch.

### Core Libraries

```python
open_source_stack = {
    "diffusers": ">=0.25.0",      # SD 1.5 pipeline, training loop
    "peft": ">=0.7.0",            # LoRA implementation
    "accelerate": ">=0.25.0",     # Multi-GPU, mixed precision
    "transformers": ">=4.36.0",   # Model architectures
    "torch": ">=2.1.0",           # Deep learning framework
}
```

**What We're NOT Reimplementing:**
- ✅ Diffusion forward/reverse process (use `diffusers`)
- ✅ LoRA adapters and merging (use `peft`)
- ✅ Distributed training (use `accelerate`)
- ✅ Attention mechanisms (use pretrained UNet)
- ✅ Noise schedulers (use `DDPMScheduler`, `DDIMScheduler`)

**What We're Customizing:**
- Custom `ClassConditionedUNet` wrapper (replaces CLIP with class embeddings)
- Custom `DefectSpectrumDataset` (from Task 1)
- Training loop configuration (hyperparameters, progressive unfreezing)

---

## 5. Hyperparameters

### LoRA Configuration

```python
lora_config = {
    "r": 8,                            # Rank (bottleneck dimension)
    "lora_alpha": 16,                  # Scaling factor (2×r)
    "lora_dropout": 0.1,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
}
```

**Trainable Parameters Breakdown:**
```python
LoRA adapters (rank=8):      ~8,000,000 params
Class embeddings (9×768):        ~7,000 params
                              ────────────────
Total trainable:              ~8,007,000 params (0.93% of 860M)
Frozen parameters:          ~851,993,000 params (99.07%)
```

### Training Hyperparameters

```python
training_config = {
    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Schedule
    "scheduler": "cosine_with_warmup",
    "warmup_steps": 100,
    "num_epochs": 100,
    
    # Batch
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    
    # Precision
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
}
```

### Class Conditioning

**Why Class Embeddings?** SD 1.5's UNet uses cross-attention that expects conditioning vectors of dimension 768 (normally from CLIP text encoder). Since we want to condition on integer class IDs (0-8) rather than text prompts, we replace the text encoder with a learnable embedding layer.

```python
class_conditioning = {
    "method": "Learnable Embedding Layer",
    "num_classes": 9,
    "embedding_dim": 768,              # Match CLIP/SD 1.5 dimension
    "guidance_scale": 7.5,
    "conditioning_dropout": 0.1,       # For classifier-free guidance
}

# Implementation:
class_embeddings = nn.Embedding(9, 768)  # Converts class_id → 768-dim vector
# class_embeddings(3) → learns to represent "cut_inner_insulation"
# class_embeddings(5) → learns to represent "good"
```

**Architecture Flow:**
```
Class ID (e.g., 3) 
    ↓
Embedding Layer → [768-dim vector] 
    ↓
UNet Cross-Attention (same as CLIP output)
    ↓
Generated Image
```

---

## 5. Fine-Tuning Strategy

### Three-Phase Progressive Training

#### **Phase 1: Frozen Backbone (Epochs 1-20)**
- **Objective:** Learn class conditioning without forgetting
- **Frozen:** VAE, UNet encoder/decoder (only LoRA adapters trainable)
- **Learning Rate:** 1e-5
- **Expected:** Loss 0.15 → 0.08

#### **Phase 2: Partial Unfreezing (Epochs 21-60)**
- **Objective:** Adapt to cable-specific features
- **Unfrozen:** UNet decoder + bottleneck (with LoRA)
- **Learning Rate:** 5e-6 (50% reduction)
- **Expected:** Loss 0.08 → 0.05, cable-specific details emerge

#### **Phase 3: Full Fine-Tuning (Epochs 61-100)**
- **Objective:** Polish generation quality
- **Unfrozen:** Full UNet (with LoRA)
- **Learning Rate:** 1e-6 (10% of Phase 1)
- **Early Stopping:** Stop if validation loss increases (overfitting)

### Validation Strategy

**Note:** No validation data split is used. Instead, validation is performed through generation and quality metrics.

#### **Generation-Based Validation (Every 5 Epochs):**
- **Sample Generation:** Generate 2-3 images per class (18-27 total)
- **Guidance Scale:** 7.5
- **Inference Steps:** 50 (DDIM)
- **Visual Inspection:** Check defect clarity, realism, class accuracy
- **Save Location:** `outputs/epoch_{epoch}/class_{id}_sample_{n}.png`

#### **FID-Based Validation (Every 10 Epochs):**
- **Metric:** Fréchet Inception Distance (FID)
- **Reference Set:** All 90 real training images
- **Generated Set:** 90 images (10 per class)
- **Target:** FID < 50 indicates good quality
- **Early Stopping:** If FID plateaus or increases for 20+ epochs

#### **Training Loss Monitoring:**
- **Primary Metric:** Training loss (denoising MSE)
- **Expected Trajectory:** 0.15 → 0.08 → 0.05 → 0.04
- **Overfitting Indicator:** Training loss continues decreasing but FID increases

**Rationale:** With only 90 samples, a traditional validation split (18 samples) produces noisy/unreliable validation loss. Generation-based validation provides more meaningful quality assessment and aligns with Task 7 evaluation requirements.

---

**Document Version:** 1.0  
**Last Updated:** January 12, 2026
