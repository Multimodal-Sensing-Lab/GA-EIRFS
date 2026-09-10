"""
Print the per-class long-tail distribution (f_b,c and f_i,c) for nuScenes.

Source: documents/Step2_nuScenes_Data_Prep.md, section 2.3.

Usage:
    python scripts/analyze_nuscenes_distribution.py \
        OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl
"""

import pickle
import sys
from collections import Counter

DEFAULT_INFOS_PATH = "OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl"


def main(infos_path: str) -> None:
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)

    instance_counts = Counter()
    frame_counts = Counter()  # frames containing each class

    for info in infos:
        names_in_frame = set(info["gt_names"])
        for name in info["gt_names"]:
            instance_counts[name] += 1
        for name in names_in_frame:
            frame_counts[name] += 1

    total_instances = sum(instance_counts.values())
    total_frames = len(infos)

    print(f"Total frames: {total_frames}")
    print(f"Total instances: {total_instances}")
    print()
    print(f"{'Class':<25} {'Instances':>10} {'Frames':>10} {'f_b,c':>8} {'f_i,c':>8}")
    print("-" * 65)
    for cls, cnt in sorted(instance_counts.items(), key=lambda x: -x[1]):
        fi_c = frame_counts[cls] / total_frames
        fb_c = cnt / total_instances
        print(f"{cls:<25} {cnt:>10,} {frame_counts[cls]:>10,} {fb_c:>8.4f} {fi_c:>8.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INFOS_PATH
    main(path)
