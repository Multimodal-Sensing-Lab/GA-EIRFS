"""
Compute and save per-class E-IRFS frequencies (f_i,c, f_b,c) for nuScenes.

Source: documents/Step2_nuScenes_Data_Prep.md, section 2.4.
Output is consumed by ga_eirfs.samplers.eirfs_sampler.

Usage:
    python scripts/compute_sampling_frequencies.py \
        OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl \
        OpenPCDet/data/nuscenes/class_frequencies.json
"""

import json
import pickle
import sys
from collections import Counter

DEFAULT_INFOS_PATH = "OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl"
DEFAULT_OUTPUT_PATH = "OpenPCDet/data/nuscenes/class_frequencies.json"


def main(infos_path: str, output_path: str) -> None:
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)

    instance_counts = Counter()
    frame_counts = Counter()

    for info in infos:
        names_in_frame = set(info["gt_names"])
        for name in info["gt_names"]:
            instance_counts[name] += 1
        for name in names_in_frame:
            frame_counts[name] += 1

    total_instances = sum(instance_counts.values())
    total_frames = len(infos)

    frequencies = {}
    for cls in instance_counts:
        frequencies[cls] = {
            "f_i_c": frame_counts[cls] / total_frames,
            "f_b_c": instance_counts[cls] / total_instances,
            "instance_count": instance_counts[cls],
            "frame_count": frame_counts[cls],
        }

    with open(output_path, "w") as f:
        json.dump(frequencies, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INFOS_PATH
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH
    main(in_path, out_path)
