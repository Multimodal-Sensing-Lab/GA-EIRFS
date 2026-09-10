"""
E-IRFS Sampler for 3D Object Detection (nuScenes + OpenPCDet)

This file implements E-IRFS adapted to 3D point cloud detection.
The formula is identical to your E-IRFS paper but applied to 3D frames
instead of 2D images, and 3D bounding box instances instead of 2D boxes.

Canonical source lives here: ga_eirfs/samplers/eirfs_sampler.py
To integrate with OpenPCDet, copy it to:
    OpenPCDet/pcdet/datasets/nuscenes/eirfs_sampler.py
and follow the integration instructions at the bottom of this file
(see also documents/Step5_OpenPCDet_Patch.md).
"""

import numpy as np
import pickle
import json
from pathlib import Path
from torch.utils.data import Sampler
from collections import Counter
import math


# ─────────────────────────────────────────────────────────────────────────────
# PART A: Frequency computation
# Call this ONCE before training to compute f_{i,c} and f_{b,c}
# ─────────────────────────────────────────────────────────────────────────────

def compute_3d_frequencies(infos_path: str) -> dict:
    """
    Compute per-class image frequency (f_{i,c}) and instance frequency (f_{b,c})
    from nuScenes OpenPCDet info files.

    These are the 3D analogues of the same quantities in your E-IRFS paper:
        f_{i,c}  = fraction of FRAMES containing at least one instance of class c
        f_{b,c}  = fraction of all bounding boxes that belong to class c

    Args:
        infos_path: path to nuscenes_infos_10sweeps_train.pkl

    Returns:
        dict mapping class_name -> {'f_i_c': float, 'f_b_c': float}
    """
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    instance_counts = Counter()
    frame_counts = Counter()

    for info in infos:
        gt_names = info.get('gt_names', [])
        names_in_frame = set(gt_names)
        for name in gt_names:
            instance_counts[name] += 1
        for name in names_in_frame:
            frame_counts[name] += 1

    total_instances = sum(instance_counts.values())
    total_frames = len(infos)

    frequencies = {}
    for cls in instance_counts:
        f_i_c = frame_counts[cls] / total_frames
        f_b_c = instance_counts[cls] / total_instances
        frequencies[cls] = {
            'f_i_c': f_i_c,
            'f_b_c': f_b_c,
            'instance_count': int(instance_counts[cls]),
            'frame_count': int(frame_counts[cls]),
        }

    return frequencies


# ─────────────────────────────────────────────────────────────────────────────
# PART B: E-IRFS repeat factor computation
# Directly implements Equation 3 from your paper, adapted for 3D frames
# ─────────────────────────────────────────────────────────────────────────────

def compute_eirfs_repeat_factors(
    frequencies: dict,
    threshold: float = 0.0001,
    alpha: float = 2.0,
) -> dict:
    """
    Compute E-IRFS class-level repeat factors.

    Formula (from E-IRFS paper, Eq. 3):
        r_c = exp( alpha * sqrt( threshold / sqrt(f_{i,c} * f_{b,c}) ) )

    In your paper for 2D:
        - f_{i,c} = image frequency = fraction of images with class c
        - f_{b,c} = box frequency   = fraction of all boxes belonging to c

    In 3D (this file):
        - f_{i,c} = frame frequency = fraction of frames with class c (same concept)
        - f_{b,c} = box frequency   = fraction of all 3D boxes belonging to c (same concept)

    The formula is IDENTICAL. Only the data source (frames vs images) differs.

    Args:
        frequencies: output of compute_3d_frequencies()
        threshold:   t in the paper; controls when oversampling activates
        alpha:       scaling parameter; higher = more aggressive oversampling

    Returns:
        dict mapping class_name -> repeat_factor (float >= 1.0)
    """
    repeat_factors = {}
    for cls, freq in frequencies.items():
        f_i_c = freq['f_i_c']
        f_b_c = freq['f_b_c']

        # geometric mean of frame and instance frequencies
        geo_mean = math.sqrt(f_i_c * f_b_c)

        # E-IRFS formula: exp(alpha * sqrt(t / geo_mean))
        # When geo_mean is small (rare class), the exponent is large -> high repeat factor
        if geo_mean > 0:
            exponent = alpha * math.sqrt(threshold / geo_mean)
            r_c = math.exp(exponent)
        else:
            r_c = float('inf')   # class with 0 instances — shouldn't happen

        repeat_factors[cls] = r_c

    return repeat_factors


def compute_frame_repeat_factors(infos: list, class_repeat_factors: dict) -> np.ndarray:
    """
    Compute per-frame repeat factor: r_i = max_{c in frame} r_c

    This is Equation after (3) in the paper:
        "Once the category-level repeat factors rc are computed,
         the image-level repeat factor ri is given by ri = max_{c∈i} rc"

    Args:
        infos: list of frame dicts (from nuscenes_infos pkl)
        class_repeat_factors: output of compute_eirfs_repeat_factors()

    Returns:
        np.ndarray of shape (N,) with r_i for each frame
    """
    frame_factors = []
    for info in infos:
        gt_names = info.get('gt_names', [])
        if len(gt_names) == 0:
            # Empty frame: no objects -> no oversampling needed
            frame_factors.append(1.0)
        else:
            # r_i = max over all classes present in this frame
            max_rc = max(
                class_repeat_factors.get(name, 1.0)
                for name in gt_names
            )
            frame_factors.append(max_rc)

    return np.array(frame_factors, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# PART C: The PyTorch Sampler
# Drop-in replacement for the default sequential sampler in OpenPCDet
# ─────────────────────────────────────────────────────────────────────────────

class EIRFSSampler3D(Sampler):
    """
    E-IRFS sampler for 3D object detection datasets.

    Implements the normalized sampling probability from Eq. 4 of the paper:
        p_i = r_i / sum_j(r_j)

    This sampler:
    1. Computes class frequencies from the training info pkl
    2. Computes E-IRFS repeat factors per class
    3. Assigns per-frame sampling probabilities
    4. At each epoch, samples len(dataset) indices according to these probabilities

    Usage:
        sampler = EIRFSSampler3D(
            infos_path='data/nuscenes/nuscenes_infos_10sweeps_train.pkl',
            threshold=0.0001,
            alpha=2.0,
        )
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)
    """

    def __init__(
        self,
        infos_path: str,
        threshold: float = 0.0001,
        alpha: float = 2.0,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.infos_path = infos_path
        self.threshold = threshold
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        # Load infos
        with open(infos_path, 'rb') as f:
            self.infos = pickle.load(f)
        self.num_samples = len(self.infos)

        # Compute frequencies and repeat factors
        self.frequencies = compute_3d_frequencies(infos_path)
        self.class_repeat_factors = compute_eirfs_repeat_factors(
            self.frequencies, threshold=threshold, alpha=alpha
        )
        self.frame_repeat_factors = compute_frame_repeat_factors(
            self.infos, self.class_repeat_factors
        )

        # Normalize to sampling probabilities: p_i = r_i / sum_j(r_j)
        total = self.frame_repeat_factors.sum()
        self.sampling_probs = self.frame_repeat_factors / total

        if self.verbose:
            self._print_summary()

    def _print_summary(self):
        print("\n" + "="*60)
        print("E-IRFS 3D Sampler Summary")
        print(f"  alpha={self.alpha}, threshold={self.threshold}")
        print(f"  Total frames: {self.num_samples}")
        print(f"\n  {'Class':<25} {'f_i,c':>8} {'f_b,c':>8} {'r_c':>10}")
        print("  " + "-"*55)
        for cls, rc in sorted(self.class_repeat_factors.items(), key=lambda x: x[1], reverse=True):
            fi = self.frequencies[cls]['f_i_c']
            fb = self.frequencies[cls]['f_b_c']
            print(f"  {cls:<25} {fi:>8.4f} {fb:>8.4f} {rc:>10.2f}")
        print("="*60 + "\n")

    def __iter__(self):
        # Sample with replacement according to E-IRFS probabilities
        # This produces exactly num_samples indices per epoch
        indices = self.rng.choice(
            self.num_samples,
            size=self.num_samples,
            replace=True,
            p=self.sampling_probs,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        # OpenPCDet's train loop calls sampler.set_epoch(cur_epoch) unconditionally
        # for any non-None sampler (mirroring torch's DistributedSampler contract).
        # self.rng already advances its own state across successive __iter__() calls
        # (one per epoch), so each epoch already draws a different sample — this is
        # a no-op that just satisfies the interface.
        pass

    def save_frequencies(self, save_path: str):
        """Save computed frequencies and repeat factors for inspection."""
        data = {
            'frequencies': self.frequencies,
            'class_repeat_factors': self.class_repeat_factors,
            'config': {'alpha': self.alpha, 'threshold': self.threshold},
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved frequency analysis to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART D: Integration into OpenPCDet
# ─────────────────────────────────────────────────────────────────────────────

"""
HOW TO PLUG THIS INTO OPENPCDET's TRAINING LOOP
================================================

OpenPCDet builds the DataLoader in:
    pcdet/datasets/__init__.py  →  build_dataloader()

The relevant function signature is:
    def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None,
                         workers=4, seed=None, logger=None, training=True, ...)

OPTION 1 (Quickest — patch build_dataloader directly):
-------------------------------------------------------
Open  pcdet/datasets/__init__.py  and find the block that creates the dataloader.
It looks roughly like:

    if training:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, ..., sampler=sampler
        )

Replace it with:

    if training:
        if dist:
            # Keep DistributedSampler for multi-GPU; E-IRFS inside each GPU shard
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        elif dataset_cfg.get('USE_EIRFS', False):
            from pcdet.datasets.nuscenes.eirfs_sampler import EIRFSSampler3D
            info_path = root_path / dataset_cfg.INFO_PATH['train'][0]
            sampler = EIRFSSampler3D(
                infos_path=str(info_path),
                threshold=dataset_cfg.get('EIRFS_THRESHOLD', 0.0001),
                alpha=dataset_cfg.get('EIRFS_ALPHA', 2.0),
                verbose=True,
            )
        else:
            sampler = None

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, ...)


OPTION 2 (Cleaner — add to the YAML config):
---------------------------------------------
In your training config yaml, add under DATA_CONFIG:

    USE_EIRFS: True
    EIRFS_ALPHA: 2.0
    EIRFS_THRESHOLD: 0.0001

Then the patch in Option 1 reads those keys automatically.


MULTI-GPU NOTE:
---------------
When training with DDP (--launcher pytorch), each GPU gets a shard of the data
via DistributedSampler. E-IRFS should run independently per GPU shard OR you
should wrap EIRFSSampler3D inside a DistributedSampler-compatible class.

The simplest approach for a research paper: train single-GPU for sampling
experiments, then verify the best configuration with multi-GPU for the full run.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PART E: Standalone test — run this to verify E-IRFS works before training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    infos_path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/nuscenes/nuscenes_infos_10sweeps_train.pkl'

    print(f"Loading infos from: {infos_path}")

    # Test frequency computation
    freqs = compute_3d_frequencies(infos_path)
    print(f"\nFound {len(freqs)} classes:")
    for cls, f in sorted(freqs.items(), key=lambda x: x[1]['f_b_c'], reverse=True):
        print(f"  {cls:<25}  f_i_c={f['f_i_c']:.4f}  f_b_c={f['f_b_c']:.4f}  "
              f"instances={f['instance_count']:,}")

    # Test repeat factor computation
    print("\nE-IRFS Repeat Factors (alpha=2.0, t=0.0001):")
    rfs = compute_eirfs_repeat_factors(freqs, threshold=0.0001, alpha=2.0)
    for cls, rc in sorted(rfs.items(), key=lambda x: -x[1]):
        print(f"  {cls:<25}  r_c = {rc:.3f}")

    # Test sampler creation
    print("\nBuilding EIRFSSampler3D...")
    sampler = EIRFSSampler3D(
        infos_path=infos_path,
        threshold=0.0001,
        alpha=2.0,
        verbose=True,
    )

    # Simulate one epoch of sampling and check class distribution
    print("Simulating one epoch of E-IRFS sampling...")
    indices = list(iter(sampler))

    # Load infos to check what classes appear
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    sampled_classes = Counter()
    for idx in indices:
        for name in infos[idx].get('gt_names', []):
            sampled_classes[name] += 1

    total_sampled = sum(sampled_classes.values())
    print(f"\nClass distribution after E-IRFS sampling (one epoch, {len(indices)} frames):")
    print(f"  {'Class':<25} {'Original %':>12} {'Sampled %':>12} {'Boost':>8}")
    print("  " + "-"*60)
    for cls in sorted(freqs.keys(), key=lambda x: -freqs[x]['f_b_c']):
        orig_pct = freqs[cls]['f_b_c'] * 100
        samp_pct = sampled_classes.get(cls, 0) / total_sampled * 100
        boost = samp_pct / orig_pct if orig_pct > 0 else 0
        print(f"  {cls:<25} {orig_pct:>11.2f}% {samp_pct:>11.2f}% {boost:>7.2f}x")

    sampler.save_frequencies('data/nuscenes/eirfs_analysis.json')
    print("\nDone. Check data/nuscenes/eirfs_analysis.json for full details.")
