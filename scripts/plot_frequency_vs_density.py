"""
Motivating figure for the paper's Introduction: class frequency vs. mean LiDAR
points per instance, on nuScenes trainval (real data, not illustrative).

The naive story would be "rare classes are also sparse" — but the actual
log-log correlation is weak (r ~= -0.16): pedestrian and traffic_cone are
common yet point-sparse, while trailer/bus are rare yet point-dense. That's
the actual motivating claim: frequency alone does not predict geometric
difficulty, so a sampling method built on frequency alone (RFS/IRFS/E-IRFS)
has no way to see the axis GA-EIRFS's G_c is built to capture. This script
does not force a misleading trend line onto data that doesn't support one.

Usage:
    python scripts/plot_frequency_vs_density.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / 'OpenPCDet/data/nuscenes/v1.0-trainval/freq_vs_points.json'
OUT_PATH = REPO_ROOT / 'output_viz/freq_vs_density.png'

MARKER_COLOR = '#2a78d6'
RARE_CLASSES = {'trailer', 'construction_vehicle', 'motorcycle', 'bicycle', 'traffic_cone'}


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')

    for cls, v in data.items():
        x, y = v['f_b_c'], v['mean_pts']
        marker = 'o' if cls in RARE_CLASSES else '^'
        ax.scatter(x, y, s=70, color=MARKER_COLOR, marker=marker,
                   edgecolors='#0b0b0b', linewidths=0.6, zorder=3)
        # nudge labels to avoid marker overlap
        offset_y = 1.12 if cls not in ('bus', 'trailer') else 0.85
        ax.annotate(cls, (x, y), xytext=(x * 1.08, y * offset_y),
                    fontsize=9, color='#0b0b0b', va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Instance frequency $f_{b,c}$ (fraction of all boxes, log scale)', fontsize=10)
    ax.set_ylabel('Mean LiDAR points per instance (log scale)', fontsize=10)
    ax.set_title('nuScenes trainval: class frequency vs. point density\n'
                  '(circle = rare class, triangle = frequent class — log-log correlation r = -0.16)',
                  fontsize=10.5)
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#e5e5e2', zorder=0)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3, color='#f2f2f0', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#c3c2b7')

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, facecolor='#ffffff')
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
