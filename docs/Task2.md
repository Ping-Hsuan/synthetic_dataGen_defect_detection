# Task 2: Model Selection

This document provides the rational of picking SD 1.5 as the pre-trained diffusion model for few-shot generation, including the hyperparameters, data configuration, and fine-tunning strategy.

---

Model choice & rationale
- `runwayml/stable-diffusion-v1-5` — Selected for a few-shot industrial-defect workflow for these practical reasons:
    - **Strong pretrained latent prior:** the checkpoint was trained on a large, diverse image dataset, which provides a robust generative prior that helps generalize from few examples (important for 20–30 samples/damage type).
    - **Latent-space efficiency:** SD v1.5 uses an Autoencoder+latent diffusion; this reduces memory and compute compared with pixel-space models, enabling 512×512 training with LoRA on a single T4/16GB GPU.
    - **Proven few-shot adaptability:** SD v1.5 is widely used with LoRA, embedding-replacement, and DreamBooth-style adapters — lowering integration risk and failure modes when adapting to defect classes.
    - **Tooling & ecosystem:** mature, well-supported libraries and adapters (e.g., `diffusers`, `peft`/LoRA, `accelerate`, ControlNet) make experiments easier to run, reproduce, and extend.
    - **Mask & inpainting readiness:** the VAE+UNet architecture supports inpainting and ControlNet-style adapters, allowing spatial mask conditioning to be added later without re-training the prior.

Data configuration
```python
data_config = {
    "product_class": "bottle",
    "dataset_path": "./scripts/Defect_Spectrum/DS-MVTec/bottle",
    "resolution": 512,
    "normalize_to_neg1_pos1": True,
    "augment": True,
    "seed": 42,
    "load_masks": False,
    "dataset_sources": ["DS-MVTec"],
    "product_classes": ["bottle"],
    "max_samples_per_damage_type": None,
    "damage_type_to_class_id": {
        "broken_large": 0,
        "broken_small": 1,
        "contamination": 2,
        "good": 3,
    },
    # Per-class counts (for reference): broken_large=20, broken_small=22, contamination=21, good=20
}
```
**Note:** SD v1.5 was trained with 512×512 images. Its VAE/UNet expect latents produced from that resolution, so using 512 preserves the learned priors and avoids distribution shift

Hyperparameters
```python
training_config = {
    "num_epochs": 250,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "lr_decay": {"epoch_100": 5e-5, "epoch_175": 2e-5},
    "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "weight_decay": 0.01},
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    },
    "class_embedding": {"num_embeddings": 4, "embedding_dim": 768, "init_std": 0.02},
    "scheduler": {"name": "DDPMScheduler", "num_train_timesteps": 1000, "validation_ddim_steps": 50},
    "seed": 42,
}
```

Fine-tuning strategy:
A) Without mask conditioned
- Overview: train UNet LoRA adapters and the `nn.Embedding` jointly. This is simple and effective for small datasets because LoRA constrains adaptation capacity.
- Steps:
    1. Initialize UNet + LoRA and `nn.Embedding` for damage-type IDs; freeze the VAE.
    2. Train LoRA + embeddings.
    3. Validate by generating fixed-seed samples per class and compute DCA / FID against the training reference.
- Provide a fast path to working conditional generation.

B) With mask-conditioned
- Goal: add explicit spatial control so the model learns "where" defects appear.
     1. Inpainting (SD v1.5)
         - Purpose: masked-region synthesis / repair where the mask indicates the area to modify.
         - Data: aligned (masked_image, mask, target) tuples at 512 resolution; apply same geometric augments to images and masks.
        - Train: load `StableDiffusionInpaintPipeline` (SD v1.5), freeze VAE, attach LoRA to the UNet, pass `masked_image` and `mask` into the pipeline (mask used as spatial conditioning), provide `nn.Embedding` via `encoder_hidden_states`, and train with the standard MSE noise-prediction loss.

     2. ControlNet-style adapter 
         - Purpose: condition generation on spatial masks without changing UNet input channels.
         - Workflow: two-stage training — (A) train `nn.Embedding` + UNet LoRA until model's class-conditional behaviors are stable; (B) freeze those weights and train the ControlNet adapter on (image, mask) pairs.
         - Benefits: separates "what" (class semantics) from "where" (spatial control), faster convergence, and lower risk of interference.