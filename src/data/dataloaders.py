from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


def collate_defect_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function optimized for diffusion-style training with optional masks.

    Produces:
      - pixel_values: FloatTensor [B,3,H,W] - images normalized to [-1, 1]
      - mask: FloatTensor [B,3,H,W] (optional) - grayscale masks normalized to [-1, 1]
      - rgb_mask: FloatTensor [B,3,H,W] (optional) - RGB masks normalized to [-1, 1]
      - meta: list[dict] - metadata including damage_type, product_class, etc.
    """
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    meta = [b.get("meta", {}) for b in batch]
    
    result = {"pixel_values": pixel_values, "meta": meta}
    
    # Stack masks if present (only include if all samples have them)
    if "mask" in batch[0]:
        masks = [b.get("mask") for b in batch if b.get("mask") is not None]
        if len(masks) == len(batch):
            result["mask"] = torch.stack(masks, dim=0)
    
    if "rgb_mask" in batch[0]:
        rgb_masks = [b.get("rgb_mask") for b in batch if b.get("rgb_mask") is not None]
        if len(rgb_masks) == len(batch):
            result["rgb_mask"] = torch.stack(rgb_masks, dim=0)
    
    return result


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """
    Create DataLoader for single-process training.

    Args:
        dataset: PyTorch Dataset (e.g., DefectSpectrumLocalDataset)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer (beneficial for CUDA)
        drop_last: Drop incomplete final batch
        prefetch_factor: Number of batches to prefetch per worker (None for default)

    """
    persistent_workers = num_workers > 0
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_defect_batch,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def make_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_replicas: int,
    rank: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = 2,
    seed: int = 0,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Create DataLoader for multi-GPU / multi-process training (PyTorch DDP).

    Args:
        dataset: PyTorch Dataset (e.g., DefectSpectrumLocalDataset)
        batch_size: Batch size PER GPU (global batch = batch_size * num_replicas)
        num_replicas: Total number of processes (typically world_size)
        rank: Rank of current process (0 to num_replicas-1)
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of worker processes per GPU
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete final batch
        prefetch_factor: Number of batches to prefetch per worker
        seed: Random seed for shuffling

    Key differences from single-GPU:
      - DistributedSampler shards dataset across ranks (no data duplication)
      - Set shuffle=False on DataLoader; sampler handles shuffling
      - MUST call sampler.set_epoch(epoch) before each epoch for proper shuffling

    Multi-GPU Performance Tips:
      - num_workers: Divide total workers by num_gpus (e.g., 8 workers / 4 GPUs = 2 per GPU)
      - batch_size: This is per-GPU batch size; effective batch = batch_size * num_replicas
      - Ensure dataset is large enough: need >> batch_size * num_replicas samples

    Example Usage:
        >>> # In each process (rank 0, 1, 2, 3 for 4-GPU training)
        >>> import torch.distributed as dist
        >>> from src.data.defect_spectrum_local import DefectSpectrumLocalDataset, LocalDatasetConfig
        >>> 
        >>> dist.init_process_group(backend="nccl")
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> 
        >>> cfg = LocalDatasetConfig(product_classes=["zipper"], load_masks=True)
        >>> dataset = DefectSpectrumLocalDataset("Defect_Spectrum", cfg)
        >>> 
        >>> loader, sampler = make_distributed_dataloader(
        ...     dataset,
        ...     batch_size=8,  # 8 per GPU → 32 total
        ...     num_replicas=world_size,
        ...     rank=rank,
        ...     num_workers=2,
        ... )
        >>> 
        >>> for epoch in range(100):
        ...     sampler.set_epoch(epoch)  # CRITICAL for proper shuffling!
        ...     for batch in loader:
        ...         # Each rank sees different data
        ...         print(f"Rank {rank}: {batch['pixel_values'].shape}")
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    persistent_workers = num_workers > 0
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sampler handles shuffling
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_defect_batch,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return loader, sampler
