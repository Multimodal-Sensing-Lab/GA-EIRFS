"""
Check whether distance-from-ego confounds the entropy and occupancy components
of G_c, the same way it was already checked for density in
scripts/check_density_range_confound.py.

For each component: measure within-class correlation between range and the
raw signal, then range-normalize and see whether the class-level ranking
(which is what G_c actually uses) changes.

Usage:
    python scripts/check_range_confound.py
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ga_eirfs.samplers.geometry_score import extract_points_in_box, compute_normal_entropy

INFOS_PATH = REPO_ROOT / 'OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl'
DATA_ROOT = REPO_ROOT / 'OpenPCDet/data/nuscenes/v1.0-trainval'
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def check_occupancy(infos, voxel_size=0.1):
    print("=" * 90)
    print("OCCUPANCY vs RANGE (cheap — uses pkl only, no point-cloud loading)")
    print("=" * 90)
    class_data = defaultdict(lambda: {'range': [], 'occ': []})
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
            class_data[name]['range'].append(r)
            class_data[name]['occ'].append(occ)

    print(f"{'class':<22}{'corr(range,occ)':>18}{'mean_occ':>12}{'pct_saturated(occ>=0.99)':>26}")
    raw_occ = {}
    for c in CLASSES:
        r = np.array(class_data[c]['range']); o = np.array(class_data[c]['occ'])
        corr = np.corrcoef(r, o)[0, 1]
        raw_occ[c] = o.mean()
        sat = (o >= 0.99).mean() * 100
        print(f"{c:<22}{corr:>18.3f}{o.mean():>12.4f}{sat:>26.1f}")

    # Regression-based correction (uses ALL data, not a narrow band): pool all
    # classes, fit occ ~ a + b*log(range), then correct each instance to a
    # reference range and re-average per class. Same methodology as the
    # density check for consistency.
    all_r = np.concatenate([np.array(class_data[c]['range']) for c in CLASSES])
    all_o = np.concatenate([np.array(class_data[c]['occ']) for c in CLASSES])
    b, a = np.polyfit(np.log(all_r), all_o, 1)  # occ = a + b*log(r)
    print(f"\nPooled linear fit: occ = {a:.4f} + ({b:.5f})*log(range)")
    R_REF = 30.0
    corrected_occ = {}
    for c in CLASSES:
        r = np.array(class_data[c]['range']); o = np.array(class_data[c]['occ'])
        o_corrected = o - b * (np.log(r) - np.log(R_REF))  # shift each instance to R_REF
        corrected_occ[c] = o_corrected.mean()

    raw_order = sorted(CLASSES, key=lambda c: raw_occ[c])
    corr_order = sorted(CLASSES, key=lambda c: corrected_occ[c])
    print("Raw occupancy ranking (sparsest/hardest first):     ", raw_order)
    print("Range-corrected occupancy ranking (sparsest first): ", corr_order)
    rho, p = spearmanr([raw_order.index(c) for c in CLASSES], [corr_order.index(c) for c in CLASSES])
    print(f"Spearman(raw rank, range-corrected rank) = {rho:.3f}, p={p:.4f}  [n=10]")
    print()


def check_entropy(infos, max_instances_per_class=300):
    print("=" * 90)
    print(f"ENTROPY vs RANGE (loads real point clouds, budgeted to {max_instances_per_class}/class)")
    print("=" * 90)
    class_data = defaultdict(lambda: {'range': [], 'ent': []})
    class_budget = defaultdict(lambda: max_instances_per_class)

    np.random.seed(42)
    shuffled = list(infos)
    np.random.shuffle(shuffled)

    for frame_idx, info in enumerate(shuffled):
        gt_names = info.get('gt_names', [])
        gt_boxes = info.get('gt_boxes', np.zeros((0, 9)))
        if len(gt_names) == 0:
            continue
        if all(class_budget[c] <= 0 for c in set(gt_names) if c in CLASSES):
            continue
        lidar_path = DATA_ROOT / info['lidar_path']
        if not lidar_path.exists():
            continue
        try:
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)
            points_xyz = points[:, :3]
        except Exception:
            continue

        for name, box in zip(gt_names, gt_boxes):
            if name not in CLASSES or class_budget[name] <= 0:
                continue
            pts_in_box = extract_points_in_box(points_xyz, box)
            if len(pts_in_box) < 5:
                continue
            ent = compute_normal_entropy(pts_in_box)
            if ent is not None:
                r = np.hypot(box[0], box[1])
                class_data[name]['range'].append(r)
                class_data[name]['ent'].append(ent)
            class_budget[name] -= 1

        if frame_idx % 1000 == 0:
            remaining = {c: b for c, b in class_budget.items() if b > 0}
            print(f"  frame {frame_idx}/{len(infos)}, remaining budget: {remaining}")
        if all(b <= 0 for b in class_budget.values()):
            break

    print(f"\n{'class':<22}{'n':>6}{'corr(range,entropy)':>22}{'mean_entropy':>14}")
    raw_ent = {}
    for c in CLASSES:
        r = np.array(class_data[c]['range']); e = np.array(class_data[c]['ent'])
        if len(r) < 5:
            print(f"{c:<22}{len(r):>6}  too few samples")
            raw_ent[c] = np.nan
            continue
        corr = np.corrcoef(r, e)[0, 1]
        raw_ent[c] = e.mean()
        print(f"{c:<22}{len(r):>6}{corr:>22.3f}{e.mean():>14.3f}")

    # Regression-based correction (uses ALL data, not a narrow band).
    valid = [c for c in CLASSES if not np.isnan(raw_ent[c])]
    all_r = np.concatenate([np.array(class_data[c]['range']) for c in valid])
    all_e = np.concatenate([np.array(class_data[c]['ent']) for c in valid])
    b, a = np.polyfit(np.log(all_r), all_e, 1)  # entropy = a + b*log(r)
    print(f"\nPooled linear fit: entropy = {a:.4f} + ({b:.5f})*log(range)  [n={len(all_r)} instances]")
    R_REF = 30.0
    corrected_ent = {}
    band_n = {}
    for c in valid:
        r = np.array(class_data[c]['range']); e = np.array(class_data[c]['ent'])
        e_corrected = e - b * (np.log(r) - np.log(R_REF))
        corrected_ent[c] = e_corrected.mean()
        band_n[c] = len(r)

    raw_order = sorted(valid, key=lambda c: raw_ent[c])
    corr_order = sorted(valid, key=lambda c: corrected_ent[c])
    print("Raw entropy ranking (lowest/simplest first):          ", raw_order)
    print("Range-corrected entropy ranking (lowest/simplest first):", corr_order)
    rho, p = spearmanr([raw_order.index(c) for c in valid], [corr_order.index(c) for c in valid])
    print(f"Spearman(raw rank, range-corrected rank) = {rho:.3f}, p={p:.4f}  [n={len(valid)}]")
    print("Sample sizes per class (all used, no band-discarding):", band_n)
    print()


if __name__ == '__main__':
    print(f"Loading infos: {INFOS_PATH}")
    with open(INFOS_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f"Loaded {len(infos)} frames\n")

    check_occupancy(infos)
    check_entropy(infos, max_instances_per_class=300)
