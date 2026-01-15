"""
DefectSpectrum local dataset loader.

Expected folder layout:
    Defect_Spectrum/
        DS-<source>/                 # e.g. DS-MVTec
            <product_class>/         # e.g. bottle, cable, zipper
                image/               # images organized by damage type
                    <damage_type>/   # e.g. broken_large, contamination, good
                mask/                # optional: per-damage-type binary masks
                    <damage_type>/
                rgb_mask/            # optional: per-damage-type RGB masks
                    <damage_type>/

What this module provides
- `LocalDatasetConfig`: configuration for resolution, augmentation, mask loading,
  class mappings, and sample limiting.
- `DefectSpectrumLocalDataset`: a PyTorch `Dataset` that scans the folder tree,
  discovers images (and optional masks), applies preprocessing and deterministic
  per-sample augmentations, and yields dictionaries with tensors and metadata.

Each sample returned by `__getitem__` contains:
- `pixel_values`: FloatTensor [3, H, W] (optionally normalized to [-1, 1])
- `class_id`: int or `None` (if `damage_type_to_class_id` mapping provided)
- `meta`: dict with `index`, `dataset_source`, `product_class`, `damage_type`,
  `filename`, `original_size`, `original_mode`
- Optional `mask` / `rgb_mask` tensors when `load_masks=True` and files exist.

Notes
- Designed to be used with the repo dataloader helpers in `src/data/dataloaders.py`
  (collate, optimized DataLoader and distributed sampler helpers).
- For reproducible augmentations across workers, set deterministic seeds and use a
  `worker_init_fn` when creating the DataLoader.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class LocalDatasetConfig:
    """Configuration for DefectSpectrum dataset loading."""
    resolution: int = 512
    augment: bool = True
    seed: int = 0
    normalize_to_neg1_pos1: bool = True
    
    # Data selection
    load_masks: bool = False  # Load masks alongside images
    dataset_sources: Optional[List[str]] = None  # ["DS-MVTec"]
    product_classes: Optional[List[str]] = None  # ["bottle"], ["pill"], ["zipper"]
    damage_types: Optional[List[str]] = None  # ["broken_large", "good"] - None means all
    max_samples_per_damage_type: Optional[int] = None  # Limit samples per damage type
    
    # Class conditioning for diffusion models
    damage_type_to_class_id: Optional[Dict[str, int]] = None  # Map damage type to class ID


class DefectSpectrumLocalDataset(Dataset[Dict[str, Any]]):
    """
    PyTorch Dataset for DefectSpectrum folder structure.
    
    Features:
    - Tracks damage types (subdirectories in image/)
    - Loads X samples per damage type
    - Optionally loads corresponding masks
    
    """
    
    def __init__(
        self,
        root_dir: str,
        cfg: Optional[LocalDatasetConfig] = None,
    ):
        self.root_dir = Path(root_dir)
        self.cfg = cfg or LocalDatasetConfig()
        
        # Build file list with metadata
        self.samples = self._build_sample_list()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root_dir} with config: {cfg}")
        
        print(f"Loaded {len(self.samples)} samples")
        self._print_statistics()
    
    def _build_sample_list(self) -> List[Dict[str, Any]]:
        """Build list of samples with metadata including damage types."""
        samples = []
        
        # Determine which dataset sources to use
        if self.cfg.dataset_sources is None:
            dataset_sources = [d.name for d in self.root_dir.iterdir() 
                             if d.is_dir() and d.name.startswith("DS-")]
        else:
            dataset_sources = self.cfg.dataset_sources
        
        # Iterate through dataset sources
        for ds_name in dataset_sources:
            ds_path = self.root_dir / ds_name
            if not ds_path.exists():
                print(f"Warning: {ds_path} does not exist, skipping")
                continue
            
            # Iterate through product classes
            for product_dir in ds_path.iterdir():
                if not product_dir.is_dir():
                    continue
                
                product_class = product_dir.name
                
                # Filter by product class if specified
                if self.cfg.product_classes is not None:
                    if product_class not in self.cfg.product_classes:
                        continue
                
                # Get image directory
                image_dir = product_dir / "image"
                if not image_dir.exists():
                    image_dir = product_dir / "images"
                    if not image_dir.exists():
                        continue
                
                # Get mask directories (if needed)
                mask_dir = product_dir / "mask"
                rgb_mask_dir = product_dir / "rgb_mask"
                if not rgb_mask_dir.exists():
                    rgb_mask_dir = product_dir / "rbg_mask"  # Handle typo
                
                # Iterate through damage type subdirectories
                for damage_dir in image_dir.iterdir():
                    if not damage_dir.is_dir():
                        continue
                    
                    damage_type = damage_dir.name
                    
                    # Skip "good" samples when masks are required (they don't have defects)
                    if self.cfg.load_masks and damage_type.lower() == "good":
                        continue
                    
                    # Filter by damage type if specified
                    if self.cfg.damage_types is not None:
                        if damage_type not in self.cfg.damage_types:
                            continue
                    
                    # Collect image files
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                    image_files = []
                    
                    for ext in image_extensions:
                        image_files.extend(damage_dir.glob(f'*{ext}'))
                        image_files.extend(damage_dir.glob(f'*{ext.upper()}'))
                    
                    # Sort for consistency
                    image_files.sort()
                    
                    # Limit samples per damage type if specified
                    if self.cfg.max_samples_per_damage_type is not None:
                        image_files = image_files[:self.cfg.max_samples_per_damage_type]
                    
                    # Add to samples with metadata
                    for img_path in image_files:
                        sample = {
                            "image_path": img_path,
                            "dataset_source": ds_name,
                            "product_class": product_class,
                            "damage_type": damage_type,
                            "filename": img_path.name,
                        }
                        
                        # Find corresponding masks if requested
                        if self.cfg.load_masks:
                            # Mask filename patterns:
                            # image: 000.png
                            # mask: 000_mask.png
                            # rgb_mask: 000_rgb_mask.png
                            img_stem = img_path.stem
                            img_ext = img_path.suffix
                            
                            mask_filename = f"{img_stem}_mask{img_ext}"
                            rgb_mask_filename = f"{img_stem}_rgb_mask{img_ext}"
                            
                            mask_path = mask_dir / damage_type / mask_filename
                            rgb_mask_path = rgb_mask_dir / damage_type / rgb_mask_filename
                            
                            if mask_path.exists():
                                sample["mask_path"] = mask_path
                            if rgb_mask_path.exists():
                                sample["rgb_mask_path"] = rgb_mask_path
                        
                        samples.append(sample)
        
        return samples
    
    def _print_statistics(self):
        """Print dataset statistics."""
        from collections import Counter
        
        dataset_sources = Counter(s["dataset_source"] for s in self.samples)
        product_classes = Counter(s["product_class"] for s in self.samples)
        damage_types = Counter(s["damage_type"] for s in self.samples)
        
        print("\nDataset sources:")
        for ds, count in dataset_sources.most_common():
            print(f"  {ds}: {count}")
        
        print("\nProduct classes:")
        for pc, count in product_classes.most_common():
            print(f"  {pc}: {count}")
        
        print("\nDamage types:")
        for dt, count in damage_types.most_common():
            print(f"  {dt}: {count}")
        
        if self.cfg.load_masks:
            with_mask = sum(1 for s in self.samples if "mask_path" in s)
            with_rgb_mask = sum(1 for s in self.samples if "rgb_mask_path" in s)
            print(f"\nMasks available: {with_mask}/{len(self.samples)}")
            print(f"RGB masks available: {with_rgb_mask}/{len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _apply_transforms(self, img: Image.Image, index: int) -> Image.Image:
        """Apply preprocessing + augmentation."""
        # Fix for grayscale masks with small value ranges (e.g., [0,1] or [0,4])
        # Binary masks should be in [0, 255] range before further processing
        if img.mode == 'L':
            img_array = np.array(img)
            max_val = img_array.max()
            # If max value is small (not already in [0, 255] range), scale it
            if max_val > 0 and max_val < 255:
                img_array = (img_array * (255.0 / max_val)).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
        
        # Ensure RGB
        img = img.convert("RGB")
        
        # Resize
        img = img.resize(
            (self.cfg.resolution, self.cfg.resolution),
            resample=Image.BICUBIC
        )
        
        if not self.cfg.augment:
            return img
        
        # Deterministic augmentation
        rng = np.random.RandomState(self.cfg.seed + index)
        
        # Random horizontal flip
        if rng.rand() < 0.5:
            img = TF.hflip(img)

        # Random rotation 15 degrees (adds variety without affecting defect visibility)
        if rng.rand() < 0.5:
            angle = rng.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        return img
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_meta = self.samples[index]
        
        # Load image
        img = Image.open(sample_meta["image_path"])
        original_size = img.size
        original_mode = img.mode
        
        # Apply transforms
        img = self._apply_transforms(img, index=index)
        
        # Convert to tensor [0, 1]
        pixel_values = TF.to_tensor(img)
        
        # Normalize to [-1, 1] if specified
        if self.cfg.normalize_to_neg1_pos1:
            pixel_values = pixel_values * 2.0 - 1.0
        
        # Get class ID for damage type (if mapping provided)
        class_id = None
        if self.cfg.damage_type_to_class_id is not None:
            damage_type = sample_meta["damage_type"]
            class_id = self.cfg.damage_type_to_class_id.get(damage_type, None)
        
        result = {
            "pixel_values": pixel_values,
            "class_id": class_id,  # Integer for class-conditional training
            "meta": {
                "index": index,
                "dataset_source": sample_meta["dataset_source"],
                "product_class": sample_meta["product_class"],
                "damage_type": sample_meta["damage_type"],
                "filename": sample_meta["filename"],
                "original_size": original_size,
                "original_mode": original_mode,
            },
        }
        
        # Load masks if requested
        if self.cfg.load_masks:
            if "mask_path" in sample_meta:
                mask = Image.open(sample_meta["mask_path"])
                mask = self._apply_transforms(mask, index=index)
                mask_tensor = TF.to_tensor(mask)
                if self.cfg.normalize_to_neg1_pos1:
                    mask_tensor = mask_tensor * 2.0 - 1.0
                result["mask"] = mask_tensor
            
            if "rgb_mask_path" in sample_meta:
                rgb_mask = Image.open(sample_meta["rgb_mask_path"])
                rgb_mask = self._apply_transforms(rgb_mask, index=index)
                rgb_mask_tensor = TF.to_tensor(rgb_mask)
                if self.cfg.normalize_to_neg1_pos1:
                    rgb_mask_tensor = rgb_mask_tensor * 2.0 - 1.0
                result["rgb_mask"] = rgb_mask_tensor
        
        return result
    
    def get_product_classes(self) -> List[str]:
        """Get list of unique product classes in dataset."""
        return sorted(set(s["product_class"] for s in self.samples))
    
    def get_dataset_sources(self) -> List[str]:
        """Get list of unique dataset sources."""
        return sorted(set(s["dataset_source"] for s in self.samples))
    
    def get_damage_types(self) -> List[str]:
        """Get list of unique damage types in dataset."""
        return sorted(set(s["damage_type"] for s in self.samples))
