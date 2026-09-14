"""
Compute and save per-class E-IRFS frequencies (f_i,c, f_b,c) for nuScenes.

Output is consumed by ga_eirfs.samplers.eirfs_sampler.

Usage:
    python scripts/compute_sampling_frequencies.py \
        OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl \
        OpenPCDet/data/nuscenes/class_frequencies.json
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ga_eirfs.samplers.eirfs_sampler import compute_3d_frequencies

DEFAULT_INFOS_PATH = "OpenPCDet/data/nuscenes/nuscenes_infos_10sweeps_train.pkl"
DEFAULT_OUTPUT_PATH = "OpenPCDet/data/nuscenes/class_frequencies.json"


def main(infos_path: str, output_path: str) -> None:
    frequencies = compute_3d_frequencies(infos_path)

    with open(output_path, "w") as f:
        json.dump(frequencies, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INFOS_PATH
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH
    main(in_path, out_path)
