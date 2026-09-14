"""
GA-EIRFS Sampler — E-IRFS base frequency term + Geometric Complexity (G_c)

Canonical source lives here: ga_eirfs/samplers/ga_eirfs_sampler.py
To integrate with OpenPCDet, copy this file alongside eirfs_sampler.py and
geometry_score.py into OpenPCDet/pcdet/datasets/nuscenes/.

── Formula ──────────────────────────────────────────────────────────────────
    r_c = exp( alpha * sqrt(threshold / geo_mean_c) * (1 + beta * G_c) )

    where geo_mean_c = sqrt(f_i,c * f_b,c)  (identical to the E-IRFS base term,
    verified against the E-IRFS IROS 2025 paper, Eq. 3)

G_c is computed once at construction from the training infos (see set_epoch
below for why it is not recomputed every epoch: a full G_c computation with
normal-entropy estimation costs on the order of ~2h on full trainval).

This sampler is the geometry-augmented E-IRFS method reported in the paper:
6 independent training runs on nuScenes and KITTI, gains concentrated in the
highest-G_c classes. See the paper for the full experimental writeup.

Usage:
    sampler = GAEIRFSSampler3D(
        infos_path='data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_train.pkl',
        data_root='data/nuscenes/v1.0-mini',
        threshold=0.01, alpha=2.0, beta=1.0,
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)
"""

import json
import math
import pickle
from collections import Counter

import numpy as np
from torch.utils.data import Sampler

from .eirfs_sampler import compute_3d_frequencies
from .geometry_score import compute_geometry_score


# ─────────────────────────────────────────────────────────────────────────────
# PART A: Geometry-Augmented Repeat Factor
# ─────────────────────────────────────────────────────────────────────────────

def compute_ga_eirfs_repeat_factors(
    frequencies: dict,
    geometry_scores: dict,
    threshold: float = 0.0001,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> dict:
    """
    r_c = exp( alpha * sqrt(threshold / geo_mean_c) * (1 + beta * G_c) )

    Args:
        frequencies: output of compute_3d_frequencies() -> {cls: {'f_i_c', 'f_b_c', ...}}
        geometry_scores: output of compute_geometry_score() -> {cls: {'G_c': float, ...}}
        threshold, alpha: same meaning as E-IRFS (see eirfs_sampler.py)
        beta: geometry weighting coefficient — how strongly G_c amplifies the exponent

    Returns:
        dict mapping class_name -> repeat_factor (float >= 1.0, roughly)
    """
    repeat_factors = {}
    for cls, freq in frequencies.items():
        f_i_c, f_b_c = freq['f_i_c'], freq['f_b_c']
        geo_mean = math.sqrt(f_i_c * f_b_c)
        if geo_mean <= 0:
            repeat_factors[cls] = float('inf')
            continue

        g_c = geometry_scores.get(cls, {}).get('G_c', 0.0)

        base_exponent = alpha * math.sqrt(threshold / geo_mean)
        exponent = base_exponent * (1.0 + beta * g_c)
        repeat_factors[cls] = math.exp(exponent)

    return repeat_factors


def compute_frame_repeat_factors(infos: list, class_repeat_factors: dict) -> np.ndarray:
    """r_i = max_{c in frame} r_c — identical rule to E-IRFS/RFS/IRFS."""
    frame_factors = []
    for info in infos:
        gt_names = info.get('gt_names', [])
        if len(gt_names) == 0:
            frame_factors.append(1.0)
        else:
            frame_factors.append(max(class_repeat_factors.get(name, 1.0) for name in gt_names))
    return np.array(frame_factors, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# PART B: The PyTorch Sampler
# ─────────────────────────────────────────────────────────────────────────────

class GAEIRFSSampler3D(Sampler):
    """
    GA-EIRFS sampler: E-IRFS base x geometry modulation.

    Two sampling modes, selected by floor_protected:

    - floor_protected=False (default, the mode used for every result in the paper):
      p_i = r_i / sum_j(r_j), a fixed-length epoch (num_samples draws) with
      probabilities renormalized to sum to 1. Because the epoch length is fixed,
      boosting some frames' probability necessarily lowers every other frame's
      *share* of that fixed pool, even frames whose own r_i never changed. This
      is the mechanism behind the trailer regression documented in the paper's
      analysis section.

    - floor_protected=True: every frame gets a guaranteed floor(r_i) copies
      (>= 1, since r_i >= 1 always) plus a proportional share of extra "bonus"
      copies, and the epoch is allowed to grow to fit them rather than
      reallocating a fixed pool. No class's exposure can ever drop below its
      own r_i-implied baseline just because another class got boosted. The
      total epoch length is fixed once at construction (not re-randomized each
      epoch) so it stays compatible with OpenPCDet's LR scheduler, which sizes
      itself from a single len(train_loader) call.
    """

    def __init__(
        self,
        infos_path: str,
        data_root: str,
        threshold: float = 0.0001,
        alpha: float = 2.0,
        beta: float = 1.0,
        geometry_scores_path: str = None,
        skip_normals: bool = False,
        max_instances_per_class: int = 300,
        floor_protected: bool = False,
        seed: int = 0,
        verbose: bool = True,
    ):
        self.infos_path = infos_path
        self.data_root = data_root
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.floor_protected = floor_protected
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        with open(infos_path, 'rb') as f:
            self.infos = pickle.load(f)
        self.num_samples = len(self.infos)

        self.frequencies = compute_3d_frequencies(infos_path)

        if geometry_scores_path is not None:
            with open(geometry_scores_path) as f:
                self.geometry_scores = json.load(f)
            if self.verbose:
                print(f'[GA-EIRFS] Loaded precomputed G_c from {geometry_scores_path}')
        else:
            if self.verbose:
                print(f'[GA-EIRFS] Computing G_c from {infos_path} '
                      f'(skip_normals={skip_normals})...')
            self.geometry_scores = compute_geometry_score(
                self.infos, data_root,
                max_instances_per_class=max_instances_per_class,
                skip_normals=skip_normals, verbose=verbose,
            )

        self._recompute()

        if self.verbose:
            self._print_summary()

    def _recompute(self):
        self.class_repeat_factors = compute_ga_eirfs_repeat_factors(
            self.frequencies, self.geometry_scores,
            threshold=self.threshold, alpha=self.alpha, beta=self.beta,
        )
        self.frame_repeat_factors = compute_frame_repeat_factors(self.infos, self.class_repeat_factors)
        total = self.frame_repeat_factors.sum()
        self.sampling_probs = self.frame_repeat_factors / total

        # Floor-protected bookkeeping (only consumed if floor_protected=True).
        # Every frame's guaranteed share is floor(r_i); the leftover fractional
        # part determines its odds of getting one bonus copy. The epoch length
        # is fixed here, once, as round(sum(r_i)) -- the expected total under
        # this scheme -- so __len__ never changes between calls.
        self._floor_counts = np.floor(self.frame_repeat_factors).astype(np.int64)
        self._frac = self.frame_repeat_factors - self._floor_counts
        self._epoch_length = int(round(self.frame_repeat_factors.sum()))
        self._num_bonus = self._epoch_length - int(self._floor_counts.sum())

    def _print_summary(self):
        print('\n' + '=' * 78)
        print('GA-EIRFS Sampler Summary')
        print(f'  alpha={self.alpha}, threshold={self.threshold}, beta={self.beta}')
        print(f'  Total frames: {self.num_samples}')
        if self.floor_protected:
            growth = 100.0 * (self._epoch_length - self.num_samples) / self.num_samples
            print(f'  floor_protected=True: epoch length {self._epoch_length} '
                  f'(+{growth:.1f}% vs. {self.num_samples} baseline), '
                  f'{int(self._num_bonus)} bonus copies distributed on top of guaranteed floors')
        else:
            print('  floor_protected=False: fixed-length epoch, probability-renormalized '
                  '(the mode used for every other result in this paper)')
        print(f"\n  {'Class':<22} {'f_i,c':>7} {'f_b,c':>7} {'G_c':>6} {'r_c':>9}")
        print('  ' + '-' * 66)
        for cls, rc in sorted(self.class_repeat_factors.items(), key=lambda x: x[1], reverse=True):
            fi = self.frequencies[cls]['f_i_c']
            fb = self.frequencies[cls]['f_b_c']
            gc = self.geometry_scores.get(cls, {}).get('G_c', 0.0)
            print(f'  {cls:<22} {fi:>7.4f} {fb:>7.4f} {gc:>6.3f} {rc:>9.3f}')
        print('=' * 78 + '\n')

    def __iter__(self):
        if not self.floor_protected:
            indices = self.rng.choice(self.num_samples, size=self.num_samples, replace=True, p=self.sampling_probs)
            return iter(indices.tolist())

        # floor_protected: guaranteed floor(r_i) copies per frame, plus exactly
        # self._num_bonus extra copies distributed (without replacement) across
        # frames weighted by their leftover fractional part. Total length is
        # fixed at self._epoch_length every epoch -- only WHICH frames get a
        # bonus copy is randomized, not how many total copies are produced.
        base = np.repeat(np.arange(self.num_samples), self._floor_counts)
        if self._num_bonus > 0 and self._frac.sum() > 0:
            bonus_p = self._frac / self._frac.sum()
            bonus = self.rng.choice(self.num_samples, size=self._num_bonus, replace=False, p=bonus_p)
            indices = np.concatenate([base, bonus])
        else:
            indices = base
        self.rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self._epoch_length if self.floor_protected else self.num_samples

    def set_epoch(self, epoch: int):
        # Satisfies the interface OpenPCDet's train loop expects of any sampler
        # (mirrors torch's DistributedSampler contract). G_c is computed once at
        # construction rather than re-derived every epoch: a full G_c
        # computation with normal-entropy estimation costs on the order of ~2h
        # on full trainval, so recomputing it every epoch is scoped out.
        pass

    def save_frequencies(self, save_path: str):
        data = {
            'frequencies': self.frequencies,
            'geometry_scores': self.geometry_scores,
            'class_repeat_factors': self.class_repeat_factors,
            'config': {'alpha': self.alpha, 'threshold': self.threshold,
                       'beta': self.beta},
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Saved GA-EIRFS analysis to {save_path}')


if __name__ == '__main__':
    import sys

    infos_path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_train.pkl'
    data_root = sys.argv[2] if len(sys.argv) > 2 else 'data/nuscenes/v1.0-mini'

    sampler = GAEIRFSSampler3D(
        infos_path=infos_path, data_root=data_root,
        threshold=0.01, alpha=2.0, beta=1.0,
        skip_normals=False, verbose=True,
    )

    print('Simulating one epoch of GA-EIRFS sampling...')
    indices = list(iter(sampler))
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)

    sampled_classes = Counter()
    for idx in indices:
        for name in infos[idx].get('gt_names', []):
            sampled_classes[name] += 1
    total_sampled = sum(sampled_classes.values())

    print(f"\n  {'Class':<22} {'Original %':>12} {'Sampled %':>12} {'Boost':>8}")
    print('  ' + '-' * 58)
    for cls in sorted(sampler.frequencies.keys(), key=lambda x: -sampler.frequencies[x]['f_b_c']):
        orig_pct = sampler.frequencies[cls]['f_b_c'] * 100
        samp_pct = sampled_classes.get(cls, 0) / total_sampled * 100
        boost = samp_pct / orig_pct if orig_pct > 0 else 0
        print(f'  {cls:<22} {orig_pct:>11.2f}% {samp_pct:>11.2f}% {boost:>7.2f}x')

    sampler.save_frequencies('data/nuscenes/v1.0-mini/ga_eirfs_analysis.json')
