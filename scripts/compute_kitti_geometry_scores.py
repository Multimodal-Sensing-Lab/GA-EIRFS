"""
Precompute G_c geometry scores for KITTI (Car/Pedestrian/Cyclist) and save to
JSON, so GA-EIRFS training doesn't recompute the ~10-20min normal-entropy pass
on every launch. Mirrors how the nuScenes GA-EIRFS runs can optionally load
precomputed scores via GA_EIRFS_GEOMETRY_SCORES in the yaml config.

Usage:
    python scripts/compute_kitti_geometry_scores.py
"""
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ga_eirfs.samplers.geometry_score import compute_geometry_score

INFOS_PATH = REPO_ROOT / 'OpenPCDet/data/kitti/kitti_infos_train_adapted.pkl'
DATA_ROOT = REPO_ROOT / 'OpenPCDet/data/kitti'
OUT_PATH = REPO_ROOT / 'OpenPCDet/data/kitti/kitti_geometry_scores.json'

if __name__ == '__main__':
    print(f'Loading {INFOS_PATH}')
    with open(INFOS_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f'Loaded {len(infos)} frames')

    geometry_scores = compute_geometry_score(
        infos, str(DATA_ROOT),
        max_instances_per_class=300,
        skip_normals=False,
        verbose=True,
        default_num_point_features=4,
    )

    print(f"\n{'class':<15}{'G_c':>8}{'density':>10}{'entropy':>10}{'occupancy':>12}")
    for cls, d in sorted(geometry_scores.items(), key=lambda x: -x[1]['G_c']):
        print(f"{cls:<15}{d['G_c']:>8.3f}{d['density_score']:>10.3f}{d['entropy_score']:>10.3f}{d['occupancy_score']:>12.3f}")

    with open(OUT_PATH, 'w') as f:
        json.dump(geometry_scores, f, indent=2)
    print(f'\nSaved: {OUT_PATH}')
