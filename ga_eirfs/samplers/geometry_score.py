"""
Geometric Complexity Score G_c for GA-EIRFS

This computes the geometry-aware difficulty signal for each class.
G_c quantifies how hard a class is to detect based on its 3D structure —
independent of how often it appears.

Run this AFTER you have the baseline and E-IRFS results.
It produces the G_c values you will use in the full GA-EIRFS sampler.

Canonical source lives here: ga_eirfs/samplers/geometry_score.py
To integrate with OpenPCDet, copy it to:
    OpenPCDet/pcdet/datasets/nuscenes/geometry_score.py

Usage:
    python ga_eirfs/samplers/geometry_score.py \
        data/nuscenes/nuscenes_infos_10sweeps_train.pkl
"""

import numpy as np
import pickle
import json
import open3d as o3d
from pathlib import Path
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1: Point Density (ρ_density_c)
# Few points per instance → harder to detect → higher score
# ─────────────────────────────────────────────────────────────────────────────

def compute_point_density_scores(infos: list) -> dict:
    """
    For each class, compute mean number of LiDAR points per instance.
    OpenPCDet stores this directly in info['num_lidar_pts'] (one value per box).

    Returns dict: class -> density_score in [0, 1]
        score=1  means very sparse (hard), score=0 means dense (easy)
    """
    class_point_counts = defaultdict(list)

    for info in infos:
        gt_names = info.get('gt_names', [])
        num_pts = info.get('num_lidar_pts', [])

        if len(gt_names) == 0 or len(num_pts) == 0:
            continue

        for name, npts in zip(gt_names, num_pts):
            class_point_counts[name].append(npts)

    # Mean points per instance per class
    mean_pts = {cls: np.mean(counts) for cls, counts in class_point_counts.items()}

    # Point counts are heavily right-skewed (e.g. nuScenes: bicycle ~5 pts vs
    # truck ~670 pts — a ~130x spread). Raw min-max normalization lets a single
    # outlier class (whichever has the most points) dominate the scale and
    # compress everyone else into a narrow band near 1.0, destroying
    # discrimination between the classes GA-EIRFS actually cares about (verified
    # on nuScenes mini: car came out at density_score=0.87, nearly as "sparse"
    # as bicycle at 1.0). log1p-transform first — standard treatment for
    # right-skewed count data — so the outlier's influence is compressed instead
    # of dominating the whole scale.
    log_pts = {cls: np.log1p(rho) for cls, rho in mean_pts.items()}
    rho_max = max(log_pts.values())
    rho_min = min(log_pts.values())

    # density_score = 1 - normalized(log(mean_pts))
    # High score = few points = harder
    density_scores = {}
    for cls, log_rho in log_pts.items():
        if rho_max > rho_min:
            normalized = (log_rho - rho_min) / (rho_max - rho_min)
        else:
            normalized = 0.5
        density_scores[cls] = 1.0 - normalized   # invert: fewer pts -> score closer to 1

    return density_scores, mean_pts


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2: Surface Normal Entropy (σ_n,c)
# High entropy = complex 3D shape = harder to detect
# ─────────────────────────────────────────────────────────────────────────────

def extract_points_in_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Extract LiDAR points inside a 3D axis-aligned bounding box.

    Args:
        points_xyz: (N, 3) LiDAR points in sensor frame
        box: [x, y, z, dx, dy, dz, heading] — box center + size + yaw

    Returns:
        (M, 3) points inside the box
    """
    cx, cy, cz, dx, dy, dz, heading = box[:7]

    # Rotate points to box frame
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    rel = points_xyz - np.array([cx, cy, cz])
    rot = np.stack([
        rel[:, 0] * cos_h - rel[:, 1] * sin_h,
        rel[:, 0] * sin_h + rel[:, 1] * cos_h,
        rel[:, 2]
    ], axis=1)

    # Filter inside half-extents
    mask = (
        (np.abs(rot[:, 0]) <= dx / 2) &
        (np.abs(rot[:, 1]) <= dy / 2) &
        (np.abs(rot[:, 2]) <= dz / 2)
    )
    return points_xyz[mask]


def compute_normal_entropy(points: np.ndarray, num_bins: int = 16):
    """
    Estimate surface normal entropy from a set of points.

    Method:
    1. Fit a local Open3D point cloud and estimate normals
    2. Build a histogram of normal directions (discretized into num_bins^2 buckets)
    3. Compute Shannon entropy of the histogram

    High entropy = normals point in many different directions = complex shape.

    Returns None (not 0.0) when there isn't enough data to estimate normals at
    all — distinct from a genuine low-entropy (simple/uniform surface) reading.
    Conflating the two systematically biases sparse-point classes (bicycle,
    pedestrian, traffic_cone — exactly the ones GA-EIRFS is meant to flag as
    hard) toward artificially LOW entropy, since most of their instances fall
    under this floor and get force-zeroed rather than actually measured
    (verified on nuScenes mini: bicycle's entropy_score came out 0.000, the
    lowest of all classes). Callers should exclude None from any class mean,
    not treat it as a measured 0.
    """
    if len(points) < 10:
        return None   # too few points to estimate normals reliably

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Estimate normals with a small radius
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )

    normals = np.asarray(pcd.normals)
    if len(normals) == 0:
        return None

    # Discretize normal directions using spherical coordinates
    # azimuth: atan2(y, x)  in [-pi, pi]
    # elevation: arcsin(z)  in [-pi/2, pi/2]
    azimuth = np.arctan2(normals[:, 1], normals[:, 0])       # [-pi, pi]
    elevation = np.arcsin(np.clip(normals[:, 2], -1, 1))     # [-pi/2, pi/2]

    az_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    el_bins = np.linspace(-np.pi / 2, np.pi / 2, num_bins + 1)

    az_idx = np.digitize(azimuth, az_bins) - 1
    el_idx = np.digitize(elevation, el_bins) - 1
    az_idx = np.clip(az_idx, 0, num_bins - 1)
    el_idx = np.clip(el_idx, 0, num_bins - 1)

    # Build 2D histogram and flatten
    hist_2d = np.zeros((num_bins, num_bins))
    for a, e in zip(az_idx, el_idx):
        hist_2d[a, e] += 1

    hist_flat = hist_2d.flatten()
    if hist_flat.sum() == 0:
        return None

    hist_prob = hist_flat / hist_flat.sum()
    # scipy entropy computes -sum(p * log(p)), base e
    return float(scipy_entropy(hist_prob + 1e-10))


def compute_normal_entropy_scores(
    infos: list,
    data_root: str,
    max_instances_per_class: int = 500,
    verbose: bool = True,
    default_num_point_features: int = 5,
) -> dict:
    """
    Compute mean surface normal entropy per class.

    Samples up to max_instances_per_class instances per class to keep
    runtime manageable (full nuScenes has 1.4M instances).

    default_num_point_features: point-cloud column width used to reshape the
    raw .bin file, when a given info doesn't specify its own
    info['num_point_features']. nuScenes' 10-sweep .bin files are
    (x,y,z,intensity,timestamp) = 5 columns; KITTI's are
    (x,y,z,intensity) = 4. Prefer the per-info field (set by
    scripts/adapt_kitti_infos.py for KITTI) over this default so mixed-dataset
    info lists reshape correctly frame-by-frame.

    Returns dict: class -> entropy_score in [0, 1]
    """
    class_entropies = defaultdict(list)
    class_instance_budget = defaultdict(lambda: max_instances_per_class)

    np.random.seed(42)
    shuffled_infos = list(infos)
    np.random.shuffle(shuffled_infos)

    for frame_idx, info in enumerate(shuffled_infos):
        gt_names = info.get('gt_names', [])
        gt_boxes = info.get('gt_boxes', np.zeros((0, 9)))

        if len(gt_names) == 0:
            continue

        # Check if any class still needs more samples
        if all(class_instance_budget[cls] <= 0 for cls in set(gt_names)):
            continue

        # Load the LiDAR point cloud
        lidar_path = Path(data_root) / info['lidar_path']
        if not lidar_path.exists():
            continue

        try:
            num_feat = info.get('num_point_features', default_num_point_features)
            points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, num_feat)
            points_xyz = points[:, :3]
        except Exception:
            continue

        for i, (name, box) in enumerate(zip(gt_names, gt_boxes)):
            if class_instance_budget[name] <= 0:
                continue

            pts_in_box = extract_points_in_box(points_xyz, box)
            if len(pts_in_box) < 5:
                continue

            ent = compute_normal_entropy(pts_in_box)
            # Budget still decrements on None (we did try this instance —
            # otherwise point-starved classes would spin through the whole
            # dataset re-trying instances that can never yield an estimate).
            # But None is excluded from the class mean, not appended as 0 —
            # see compute_normal_entropy's docstring for why that distinction
            # matters here specifically.
            if ent is not None:
                class_entropies[name].append(ent)
            class_instance_budget[name] -= 1

        if verbose and frame_idx % 500 == 0:
            remaining = {cls: b for cls, b in class_instance_budget.items() if b > 0}
            print(f"  Frame {frame_idx}/{len(infos)}, remaining budget: {remaining}")

        # Stop early if all classes are saturated
        if all(b <= 0 for b in class_instance_budget.values()):
            break

    # Normalize to [0, 1]
    mean_entropies = {cls: np.mean(ents) for cls, ents in class_entropies.items()}
    if len(mean_entropies) == 0:
        return {}

    ent_max = max(mean_entropies.values())
    ent_min = min(mean_entropies.values())

    entropy_scores = {}
    for cls, ent in mean_entropies.items():
        if ent_max > ent_min:
            entropy_scores[cls] = (ent - ent_min) / (ent_max - ent_min)
        else:
            entropy_scores[cls] = 0.5

    return entropy_scores, mean_entropies


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3: Voxel Occupancy Sparsity (η_occ_c)
# Low occupancy = sparse 3D structure = harder to detect
# ─────────────────────────────────────────────────────────────────────────────

def compute_voxel_occupancy_scores(
    infos: list,
    voxel_size: float = 0.1,
) -> dict:
    """
    Compute voxel occupancy ratio per class.
    η_occ = 1 - (occupied voxels / reference voxel budget)

    High score = sparse occupancy = harder to detect.

    The reference budget is the box's SURFACE area in voxel units
    (2*(dx*dy + dy*dz + dx*dz)) / voxel_size^2, not its volume. A LiDAR only
    ever returns a thin shell off the visible surface of an object, never its
    interior — so point count scales with surface area (~length^2), not
    volume (~length^3). Dividing by volume systematically penalizes large
    objects: verified on nuScenes mini, trailer/bus (large, well-observed
    vehicles) came out with the highest occupancy "sparsity" scores of any
    class purely from their bounding-box size, not from any real structural
    complexity — the metric was conflating object size with geometric hardness.
    """
    class_occupancies = defaultdict(list)

    for info in infos:
        gt_names = info.get('gt_names', [])
        gt_boxes = info.get('gt_boxes', np.zeros((0, 9)))
        num_pts = info.get('num_lidar_pts', [])

        if len(gt_names) == 0:
            continue

        for name, box, npts in zip(gt_names, gt_boxes, num_pts):
            dx, dy, dz = box[3], box[4], box[5]
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue

            # Reference voxel budget: box surface area, not volume (see docstring).
            surface_area = 2 * (dx * dy + dy * dz + dx * dz)
            total_voxels = surface_area / (voxel_size ** 2)
            if total_voxels < 1:
                continue

            # Approximate: assume each point occupies one surface voxel
            occupied_voxels = min(npts, total_voxels)
            occupancy_ratio = occupied_voxels / total_voxels
            class_occupancies[name].append(occupancy_ratio)

    mean_occupancy = {cls: np.mean(occ) for cls, occ in class_occupancies.items()}

    occ_max = max(mean_occupancy.values())
    occ_min = min(mean_occupancy.values())

    occupancy_scores = {}
    for cls, occ in mean_occupancy.items():
        if occ_max > occ_min:
            normalized = (occ - occ_min) / (occ_max - occ_min)
        else:
            normalized = 0.5
        occupancy_scores[cls] = 1.0 - normalized   # invert: sparse -> score closer to 1

    return occupancy_scores, mean_occupancy


# ─────────────────────────────────────────────────────────────────────────────
# Combined G_c score
# ─────────────────────────────────────────────────────────────────────────────

def compute_geometry_score(
    infos: list,
    data_root: str,
    lambda_density: float = 0.5,
    lambda_entropy: float = 0.3,
    lambda_occupancy: float = 0.2,
    max_instances_per_class: int = 300,
    skip_normals: bool = False,
    verbose: bool = True,
    default_num_point_features: int = 5,
) -> dict:
    """
    Compute the combined Geometric Complexity Score G_c for each class.

    G_c = λ_1 * ρ_density_c + λ_2 * σ_n,c + λ_3 * η_occ_c

    All three components are in [0, 1], so G_c is in [0, 1].

    Args:
        skip_normals: set True to skip the expensive normal entropy computation
                      (useful for a quick first run; set False for the paper)
    """
    assert abs(lambda_density + lambda_entropy + lambda_occupancy - 1.0) < 1e-6, \
        "Lambda weights must sum to 1"

    if verbose:
        print("Computing point density scores...")
    density_scores, mean_pts = compute_point_density_scores(infos)

    if not skip_normals:
        if verbose:
            print("\nComputing surface normal entropy scores (this takes ~10–20 min)...")
        entropy_scores, mean_entropies = compute_normal_entropy_scores(
            infos, data_root, max_instances_per_class=max_instances_per_class, verbose=verbose,
            default_num_point_features=default_num_point_features,
        )
    else:
        entropy_scores = {cls: 0.5 for cls in density_scores}
        mean_entropies = {}
        if verbose:
            print("Skipping normal entropy (skip_normals=True)")

    if verbose:
        print("\nComputing voxel occupancy scores...")
    occupancy_scores, mean_occupancy = compute_voxel_occupancy_scores(infos)

    # Combine all three signals
    all_classes = set(density_scores) | set(entropy_scores) | set(occupancy_scores)
    geometry_scores = {}
    for cls in all_classes:
        rho = density_scores.get(cls, 0.5)
        sigma = entropy_scores.get(cls, 0.5)
        eta = occupancy_scores.get(cls, 0.5)
        G_c = lambda_density * rho + lambda_entropy * sigma + lambda_occupancy * eta
        geometry_scores[cls] = {
            'G_c': G_c,
            'density_score': rho,
            'entropy_score': sigma,
            'occupancy_score': eta,
            'mean_pts_per_instance': float(mean_pts.get(cls, 0)),
            'mean_normal_entropy': float(mean_entropies.get(cls, 0)),
            'mean_occupancy_ratio': float(mean_occupancy.get(cls, 0)),
        }

    return geometry_scores


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute and print G_c for all nuScenes classes
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    infos_path = sys.argv[1] if len(sys.argv) > 1 else \
        'data/nuscenes/nuscenes_infos_10sweeps_train.pkl'
    data_root = sys.argv[2] if len(sys.argv) > 2 else 'data/nuscenes'

    print(f"Loading infos: {infos_path}")
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    print(f"Loaded {len(infos)} frames")

    # Skip normals on first run for speed; enable for paper
    scores = compute_geometry_score(
        infos, data_root,
        lambda_density=0.5, lambda_entropy=0.3, lambda_occupancy=0.2,
        max_instances_per_class=300,
        skip_normals=False,   # set True for a quick test
        verbose=True,
    )

    print("\n" + "="*75)
    print("Geometric Complexity Scores G_c per class")
    print(f"{'Class':<25} {'G_c':>6} {'Density':>8} {'Entropy':>8} {'Occupancy':>10} {'MeanPts':>8}")
    print("-"*75)
    for cls, v in sorted(scores.items(), key=lambda x: -x[1]['G_c']):
        print(f"{cls:<25} {v['G_c']:>6.3f} {v['density_score']:>8.3f} "
              f"{v['entropy_score']:>8.3f} {v['occupancy_score']:>10.3f} "
              f"{v['mean_pts_per_instance']:>8.1f}")
    print("="*75)
    print()
    print("Expected ordering (hypothesis):")
    print("  bicycle > motorcycle > traffic_cone > pedestrian > trailer > car")
    print("  (complex thin geometry > simple box geometry)")

    # Save for use in GA-EIRFS sampler
    save_path = 'data/nuscenes/geometry_scores.json'
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved to {save_path}")
