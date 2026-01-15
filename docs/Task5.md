# Mask-Guided Diffusion for Few-Shot Industrial Defect Generation

This document describes two segmentation-mask–based conditioning strategies
for few-shot industrial defect synthesis using diffusion models:
**Diffusion Inpainting** and **ControlNet-based conditioning**.

Both methods leverage pixel-level defect masks to improve controllability,
sample efficiency, and realism in low-data industrial inspection settings.

---

## Problem Setting

- Dataset provides:
  - RGB images of industrial products
  - Pixel-level defect segmentation masks (defective samples only)
  - Discrete damage-type labels (e.g., `broken_large`, `broken_little`, `contamination`)
- Clean (`good`) samples contain **no defect mask**
- Only a small number of examples are available per defect type

The goal is to generate realistic synthetic defect images while preserving
product geometry and background, enabling data augmentation for defect detection models.

---

## Conditioning Signals

| Signal | Purpose |
|------|--------|
| Damage-type embedding | Controls **what defect** to generate |
| Segmentation mask | Controls **where defect** appears |
| Image context | Preserves product geometry and background |

As established in Task 4, we fine-tune a pretrained Stable Diffusion 1.5 model
using a learned damage-type embedding in place of the standard CLIP text
conditioning. This embedding is injected into the diffusion U-Net via the
standard cross-attention layers, while Low-Rank Adaptation (LoRA) is used to
efficiently fine-tune the U-Net with the base model weights kept frozen.

---

## Method 1: Diffusion Inpainting (SD 1.5 Inpainting)

### Core Idea

Diffusion inpainting constrains the generative process to **only regenerate
pixels inside a predefined defect region**, while keeping all other pixels
explicitly fixed.

This is achieved by modifying the U-Net input to include both the defect mask
and the masked image as additional input channels.

---

### Model Architecture

- Base model: **Stable Diffusion 1.5 Inpainting checkpoint**
- U-Net input channels: **9**
  - 4 × noisy image latents
  - 4 × masked image latents
  - 1 × binary defect mask
- VAE: frozen
- Base U-Net weights: frozen
- Trainable parameters:
  - LoRA adapters on U-Net attention layers
  - Damage-type embedding table

No manual modification of the U-Net architecture is required when using the
pretrained inpainting checkpoint.

---

### Inpainting Mechanics

Given:
- original image `I`
- defect mask `M` (1 = defect region, 0 = background)

Steps:
1. Mask the image:  
   `I_masked = I * (1 - M)`
2. Encode both `I` and `I_masked` using the VAE
3. Add noise **only inside the defect region**
4. Concatenate inputs and pass to the inpainting U-Net

Effect:
- Pixels outside the mask are re-injected at every diffusion step
- The model is physically unable to modify background or product geometry
- Diffusion reconstructs content **only inside the defect region**

This acts as a **hard pixel-level constraint**, not a learned one.

**Note on the `good` damage type:** Samples labeled as good contain no defects and therefore have no associated segmentation mask. In these cases, inpainting is bypassed

---

### Strengths of Inpainting

- Strong spatial localization
- Guaranteed preservation of product geometry
- Minimal hallucination outside defect regions
- Simple and stable training pipeline

---

### Limitations

- Requires a base image to edit
- Limited flexibility in defect placement
- Less suitable for free-form full-image generation

---

## Method 2: ControlNet with Segmentation Masks

### Core Idea

ControlNet introduces **feature-level spatial conditioning** by processing
segmentation masks through a parallel neural network that guides the diffusion
process internally, without modifying the U-Net input channels.

Unlike inpainting, ControlNet does not impose hard pixel constraints.

---

### Model Architecture

- Base model: Stable Diffusion 1.5
- ControlNet:
  - Parallel network with the same architecture as the U-Net
  - Receives the segmentation mask as input
  - Produces spatial feature guidance
- Base U-Net weights: frozen
- Trainable parameters:
  - ControlNet weights
  - Optional LoRA adapters on U-Net
  - Damage-type embedding table

---

### ControlNet Conditioning Mechanism

1. The segmentation mask is encoded by ControlNet
2. ControlNet outputs feature offsets at multiple U-Net layers
3. These offsets are injected into the frozen diffusion model
4. The diffusion process is guided to respect the spatial structure of the mask

Effect:
- Mask influences **how features evolve**, not just where pixels change
- The entire image can be generated or modified
- Defect shape, size, and placement can vary more freely

This acts as a **soft structural constraint**.

---

### Strengths of ControlNet

- Strong control over defect shape and topology
- Supports full-image generation from noise
- Enables defect placement at new locations
- Higher diversity and flexibility

---

### Limitations

- More complex training pipeline
- Higher computational cost
- Outside-mask regions are influenced, not frozen

---

## Mask-Aware Geometric Augmentation (Shared)

To improve diversity and generalization in the few-shot regime, **geometric
augmentations** are applied consistently to both the image and its
segmentation mask.

Supported augmentations:
- Horizontal / vertical flips (when physically plausible)
- Rotation (e.g., ±10–30°)
- Translation and random cropping
- Mild scaling

Photometric augmentations are intentionally excluded to preserve defect
appearance semantics.

---

## Inpainting vs ControlNet: Summary

| Aspect | Inpainting | ControlNet |
|------|-----------|-----------|
| Mask role | Hard pixel constraint | Soft feature guidance |
| Outside-mask pixels | Frozen | Influenced |
| Requires base image | Yes | No |
| Defect placement flexibility | Low | High |
| Sample efficiency | Very high | High |
| Implementation complexity | Low | Medium |

---

## Design Choice in This Project

Given:
- Accurate defect segmentation masks
- Availability of real product images
- Strong requirement to preserve product geometry
- Few-shot constraints

**Diffusion inpainting is used as the primary method**, as it provides strong
localization and stability with minimal complexity.

**ControlNet is considered a natural extension** when increased diversity,
shape control, or full-image generation is required.

---

## Key Takeaway

Segmentation masks allow diffusion models to decouple:
- *what defect to generate* (damage-type embedding)
- *where the defect appears* (mask-based conditioning)

This decoupling is critical for controllable, realistic, and sample-efficient
synthetic defect generation in industrial inspection tasks.
