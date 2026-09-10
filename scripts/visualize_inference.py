"""
Run a trained OpenPCDet checkpoint on one nuScenes val sample and render a
bird's-eye-view (BEV) figure: LiDAR points + ground-truth boxes (green) +
predicted boxes (red, with class/score labels).

Headless-friendly (matplotlib, no Open3D/Mayavi window) — unlike OpenPCDet's own
tools/demo.py, which only supports an interactive 3D viewer. Also directly useful
for the paper: proposal section 4.5 calls for exactly this kind of qualitative
point-cloud figure (baseline vs. GA-EIRFS on rare-class detections).

Usage (run from anywhere; OPENPCDET_ROOT below can be overridden):
    python scripts/visualize_inference.py \
        --cfg_file tools/cfgs/nuscenes_models/centerpoint_mini.yaml \
        --ckpt output/nuscenes_models/centerpoint_mini/baseline_mini/ckpt/checkpoint_epoch_5.pth \
        --sample_idx 0 \
        --out output_viz/mini_inference_sample0.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_ROOT = REPO_ROOT / 'OpenPCDet'
sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


def box_corners_2d(cx, cy, dx, dy, heading):
    """4 BEV corners of a rotated box, in order for closing the polygon."""
    hx, hy = dx / 2, dy / 2
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([cx, cy])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/nuscenes_models/centerpoint_mini.yaml')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--point_range', type=float, default=50.0, help='+/- meters shown around the crop center')
    parser.add_argument('--center_x', type=float, default=0.0, help='crop window center, ego frame (default: ego position)')
    parser.add_argument('--center_y', type=float, default=0.0, help='crop window center, ego frame (default: ego position)')
    parser.add_argument('--out', type=str, default='output_viz/mini_inference_sample.png')
    args = parser.parse_args()

    cfg_file = OPENPCDET_ROOT / args.cfg_file
    ckpt = OPENPCDET_ROOT / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # cd into tools/ first: both the model cfg's _BASE_CONFIG_ pointer and
    # DATA_CONFIG.DATA_PATH ('../data/nuscenes') are relative to that directory,
    # same as when train.py/test.py run from there.
    import os
    os.chdir(OPENPCDET_ROOT / 'tools')

    cfg_from_yaml_file(str(cfg_file), cfg)
    logger = common_utils.create_logger()

    val_set, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, training=False, logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_set)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    data_dict = val_set.collate_batch([val_set[args.sample_idx]])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    points = data_dict['points'].cpu().numpy()[:, 1:4]  # drop batch idx col, keep x,y,z
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

    keep = pred_scores >= args.score_thresh
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]

    gt_boxes = None
    if 'gt_boxes' in data_dict:
        gt = data_dict['gt_boxes'][0].cpu().numpy()
        gt_boxes = gt[np.any(gt != 0, axis=1)]  # drop zero-padding rows

    class_names = cfg.CLASS_NAMES
    r = args.point_range
    cx0, cy0 = args.center_x, args.center_y

    # Boxes outside the display window get clipped out here too, not just points —
    # otherwise a GT/pred box's text label (matplotlib Text has clip_on=False by
    # default) renders at its real, off-screen data coordinate and ends up floating
    # over the title/figure margin instead of being cleanly cropped out.
    if gt_boxes is not None:
        gt_in_range = (np.abs(gt_boxes[:, 0] - cx0) <= r) & (np.abs(gt_boxes[:, 1] - cy0) <= r)
        gt_boxes = gt_boxes[gt_in_range]
    pred_in_range = (np.abs(pred_boxes[:, 0] - cx0) <= r) & (np.abs(pred_boxes[:, 1] - cy0) <= r)
    pred_boxes, pred_scores, pred_labels = pred_boxes[pred_in_range], pred_scores[pred_in_range], pred_labels[pred_in_range]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    in_range = (np.abs(points[:, 0] - cx0) <= r) & (np.abs(points[:, 1] - cy0) <= r)
    pts = points[in_range]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=pts[:, 2], cmap='viridis', alpha=0.6, linewidths=0)

    if gt_boxes is not None:
        for box in gt_boxes:
            cx, cy, dx, dy, heading, cls_idx = box[0], box[1], box[3], box[4], box[6], int(box[-1])
            corners = box_corners_2d(cx, cy, dx, dy, heading)
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='lime', linewidth=1.6))
            name = class_names[cls_idx - 1] if 1 <= cls_idx <= len(class_names) else str(cls_idx)
            ax.text(cx, cy, name, color='lime', fontsize=6, ha='center', va='bottom')

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        cx, cy, dx, dy, heading = box[0], box[1], box[3], box[4], box[6]
        corners = box_corners_2d(cx, cy, dx, dy, heading)
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor='red', linewidth=1.6, linestyle='--'))
        name = class_names[label - 1] if 1 <= label <= len(class_names) else str(label)
        ax.text(cx, cy, f'{name} {score:.2f}', color='red', fontsize=6, ha='center', va='top')

    ax.set_xlim(cx0 - r, cx0 + r)
    ax.set_ylim(cy0 - r, cy0 + r)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlabel('x (m, ego frame)')
    ax.set_ylabel('y (m, ego frame)')
    n_gt = 0 if gt_boxes is None else len(gt_boxes)
    ax.set_title(
        f'CenterPoint BEV — sample {args.sample_idx} — '
        f'GT (green): {n_gt}  Pred (red, score>={args.score_thresh}): {len(pred_boxes)}'
    )
    ax.plot(0, 0, marker='^', color='white', markersize=10)  # ego vehicle marker

    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    print(f'GT boxes: {n_gt} | Predicted boxes (score>={args.score_thresh}): {len(pred_boxes)}')
    if len(pred_boxes):
        for score, label in sorted(zip(pred_scores, pred_labels), reverse=True)[:10]:
            name = class_names[label - 1] if 1 <= label <= len(class_names) else str(label)
            print(f'  {name:<20} score={score:.3f}')


if __name__ == '__main__':
    main()
