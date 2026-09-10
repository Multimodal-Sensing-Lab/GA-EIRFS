"""
Density-vs-range confound check, and the final range-corrected G_c comparison.

This persists the exact computation originally run inline during paper-writing
(never saved to a script at the time) so it's independently reproducible.
Companion to scripts/check_range_confound.py, which already persists the
entropy/occupancy vs. range checks.

Methodology note (important for matching the paper's exact wording): the
final "range-corrected G_c" comparison below only range-corrects the entropy
and occupancy components. Density's own robustness to range was established
separately (raw ranking vs. range-normalized-instance ranking, both computed
directly from the pkl, no dependence on the pooled regression used here) and
found robust (Spearman rho~0.976) -- so the final combined-G_c comparison
reuses the standard (uncorrected) density score unchanged, exactly as
compute_point_density_scores() already computes it. It does NOT feed a
range-corrected density into the final G_c number. The paper text must say
"entropy and occupancy" range-corrected in the final combination, not
"all three" -- if it says "all three," that's a bug in the prose, not in
this computation.

Usage:
    python scripts/check_density_range_confound.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ga_eirfs.samplers.geometry_score import (
    extract_points_in_box, compute_normal_entropy, compute_point_density_scores,
)

INFOS_PATH = REPO_ROOT / 'OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl'
DATA_ROOT = REPO_ROOT / 'OpenPCDet/data/nuscenes/v1.0-trainval'
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
R_REF = 30.0


def check_density_vs_range(infos):
    print("=" * 90)
    print("DENSITY vs RANGE (pooled log-log fit across all instances)")
    print("=" * 90)
    class_data = {c: {'range': [], 'pts': []} for c in CLASSES}
    for info in infos:
        for name, box, n in zip(info['gt_names'], info['gt_boxes'], info['num_lidar_pts']):
            if name in class_data:
                r = np.hypot(box[0], box[1])
                class_data[name]['range'].append(r)
                class_data[name]['pts'].append(n)

    print(f"{'class':<22}{'mean_range(m)':>15}{'mean_pts':>12}{'corr(log_r,log_pts+1)':>24}")
    for c in CLASSES:
        r = np.array(class_data[c]['range']); p = np.array(class_data[c]['pts'])
        valid = p > 0
        corr_log = np.corrcoef(np.log(r[valid]), np.log(p[valid] + 1))[0, 1]
        print(f"{c:<22}{r.mean():>15.2f}{p.mean():>12.1f}{corr_log:>24.3f}")

    all_r = np.concatenate([np.array(class_data[c]['range']) for c in CLASSES])
    all_p = np.concatenate([np.array(class_data[c]['pts']) for c in CLASSES])
    valid = all_p > 0
    slope, intercept = np.polyfit(np.log(all_r[valid]), np.log(all_p[valid] + 1), 1)
    print(f"\npooled log-log slope (expect ~-2 if pure 1/r^2 falloff): {slope:.3f}")

    # Range-normalize each instance's point count to R_REF, then re-average per class,
    # to check whether the CLASS-LEVEL ranking (not the raw instance-level noise) is
    # robust to range -- this is the check that feeds the paper's claim, not the
    # final G_c combination below.
    raw_density = {}
    norm_density = {}
    for c in CLASSES:
        r = np.array(class_data[c]['range']); p = np.array(class_data[c]['pts'])
        raw_density[c] = p.mean()
        p_norm = p * (r / R_REF) ** (-slope)
        norm_density[c] = p_norm.mean()

    raw_order = sorted(CLASSES, key=lambda c: raw_density[c])
    norm_order = sorted(CLASSES, key=lambda c: norm_density[c])
    rho, p_val = spearmanr([raw_order.index(c) for c in CLASSES], [norm_order.index(c) for c in CLASSES])
    print(f"Spearman(raw density class ranking, range-normalized ranking) = {rho:.3f}, p={p_val:.4f}")
    print()
    return slope


def recompute_entropy_and_occupancy_corrections(infos, max_instances_per_class=300):
    """Reproduces the pooled regression correction already persisted in
    check_range_confound.py, returning per-class corrected entropy/occupancy
    means (not just the rho check) so they can feed the final G_c comparison."""
    from collections import defaultdict

    # Occupancy (cheap, uses pkl only)
    occ_data = defaultdict(lambda: {'range': [], 'occ': []})
    voxel_size = 0.1
    for info in infos:
        for name, box, npts in zip(info['gt_names'], info['gt_boxes'], info['num_lidar_pts']):
            if name not in CLASSES:
                continue
            dx, dy, dz = box[3], box[4], box[5]
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            surface_area = 2 * (dx * dy + dy * dz + dx * dz)
            total_voxels = surface_area / (voxel_size ** 2)
            if total_voxels < 1:
                continue
            occ = min(npts, total_voxels) / total_voxels
            r = np.hypot(box[0], box[1])
            occ_data[name]['range'].append(r)
            occ_data[name]['occ'].append(occ)

    all_r = np.concatenate([np.array(occ_data[c]['range']) for c in CLASSES])
    all_o = np.concatenate([np.array(occ_data[c]['occ']) for c in CLASSES])
    b_occ, a_occ = np.polyfit(np.log(all_r), all_o, 1)
    corrected_occ_mean = {}
    for c in CLASSES:
        r = np.array(occ_data[c]['range']); o = np.array(occ_data[c]['occ'])
        corrected_occ_mean[c] = (o - b_occ * (np.log(r) - np.log(R_REF))).mean()

    # Entropy (needs real point clouds, budgeted sampling -- same seed/budget as before)
    ent_data = defaultdict(lambda: {'range': [], 'ent': []})
    budget = defaultdict(lambda: max_instances_per_class)
    np.random.seed(42)
    shuffled = list(infos)
    np.random.shuffle(shuffled)
    import os
    for info in shuffled:
        names = info.get('gt_names', [])
        if len(names) == 0:
            continue
        if all(budget[c] <= 0 for c in set(names) if c in CLASSES):
            continue
        lp = str(DATA_ROOT / info['lidar_path'])
        if not os.path.exists(lp):
            continue
        try:
            pts = np.fromfile(lp, dtype=np.float32).reshape(-1, 5)[:, :3]
        except Exception:
            continue
        for name, box in zip(names, info['gt_boxes']):
            if name not in CLASSES or budget[name] <= 0:
                continue
            pib = extract_points_in_box(pts, box)
            if len(pib) < 5:
                continue
            ent = compute_normal_entropy(pib)
            if ent is not None:
                r = np.hypot(box[0], box[1])
                ent_data[name]['range'].append(r)
                ent_data[name]['ent'].append(ent)
            budget[name] -= 1
        if all(b <= 0 for b in budget.values()):
            break

    all_r_e = np.concatenate([np.array(ent_data[c]['range']) for c in CLASSES])
    all_e_e = np.concatenate([np.array(ent_data[c]['ent']) for c in CLASSES])
    b_ent, a_ent = np.polyfit(np.log(all_r_e), all_e_e, 1)
    corrected_ent_mean = {}
    for c in CLASSES:
        r = np.array(ent_data[c]['range']); e = np.array(ent_data[c]['ent'])
        corrected_ent_mean[c] = (e - b_ent * (np.log(r) - np.log(R_REF))).mean()

    return corrected_ent_mean, corrected_occ_mean


def final_combined_check(infos):
    print("=" * 90)
    print("FINAL COMBINED G_c: entropy+occupancy range-corrected, density UNCHANGED")
    print("(density is not re-corrected here -- its own robustness was already")
    print(" established above via the raw-vs-range-normalized ranking check)")
    print("=" * 90)

    density_scores, mean_pts = compute_point_density_scores(infos)
    corrected_ent_mean, corrected_occ_mean = recompute_entropy_and_occupancy_corrections(infos)

    occ_max, occ_min = max(corrected_occ_mean.values()), min(corrected_occ_mean.values())
    occupancy_scores_corrected = {c: 1.0 - (corrected_occ_mean[c] - occ_min) / (occ_max - occ_min) for c in CLASSES}

    ent_max, ent_min = max(corrected_ent_mean.values()), min(corrected_ent_mean.values())
    entropy_scores_corrected = {c: (corrected_ent_mean[c] - ent_min) / (ent_max - ent_min) for c in CLASSES}

    published_Gc = {'bicycle': 0.823, 'pedestrian': 0.807, 'motorcycle': 0.718, 'car': 0.601,
                     'traffic_cone': 0.588, 'construction_vehicle': 0.518, 'truck': 0.431,
                     'bus': 0.342, 'trailer': 0.337, 'barrier': 0.252}

    Gc_corrected = {}
    for c in CLASSES:
        Gc_corrected[c] = 0.5 * density_scores[c] + 0.3 * entropy_scores_corrected[c] + 0.2 * occupancy_scores_corrected[c]

    print(f"{'class':<22}{'published_Gc':>14}{'range_corrected_Gc':>20}")
    for c in sorted(CLASSES, key=lambda c: -Gc_corrected[c]):
        print(f"{c:<22}{published_Gc[c]:>14.3f}{Gc_corrected[c]:>20.3f}")

    pub_order = sorted(CLASSES, key=lambda c: -published_Gc[c])
    corr_order = sorted(CLASSES, key=lambda c: -Gc_corrected[c])
    rho, p = spearmanr([pub_order.index(c) for c in CLASSES], [corr_order.index(c) for c in CLASSES])
    print(f"\nSpearman(published G_c rank, range-corrected G_c rank) = {rho:.3f}, p={p:.4f}")
    print("Published G_c top-2:", pub_order[:2])
    print("Range-corrected G_c top-2:", corr_order[:2])
    print(f"bicycle: {published_Gc['bicycle']:.3f} -> {Gc_corrected['bicycle']:.3f}")
    print(f"motorcycle: {published_Gc['motorcycle']:.3f} -> {Gc_corrected['motorcycle']:.3f}")


if __name__ == '__main__':
    print(f"Loading infos: {INFOS_PATH}")
    with open(INFOS_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f"Loaded {len(infos)} frames\n")

    check_density_vs_range(infos)
    final_combined_check(infos)
