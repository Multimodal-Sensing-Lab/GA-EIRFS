"""
Print the per-class long-tail distribution (f_b,c and f_i,c) for nuScenes.

Usage:
    python scripts/analyze_nuscenes_distribution.py \
        OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl
"""

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ga_eirfs.samplers.eirfs_sampler import compute_3d_frequencies

DEFAULT_INFOS_PATH = "OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl"


def main(infos_path: str) -> None:
    with open(infos_path, "rb") as f:
        total_frames = len(pickle.load(f))

    frequencies = compute_3d_frequencies(infos_path)
    total_instances = sum(f["instance_count"] for f in frequencies.values())

    print(f"Total frames: {total_frames}")
    print(f"Total instances: {total_instances}")
    print()
    print(f"{'Class':<25} {'Instances':>10} {'Frames':>10} {'f_b,c':>8} {'f_i,c':>8}")
    print("-" * 65)
    for cls, f in sorted(frequencies.items(), key=lambda x: -x[1]["instance_count"]):
        print(f"{cls:<25} {f['instance_count']:>10,} {f['frame_count']:>10,} "
              f"{f['f_b_c']:>8.4f} {f['f_i_c']:>8.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INFOS_PATH
    main(path)
