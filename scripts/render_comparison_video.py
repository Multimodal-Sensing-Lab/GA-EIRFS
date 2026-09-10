"""
Render a video (mp4) of consecutive nuScenes val keyframes from one 20s scene,
as a 3-panel side-by-side BEV comparison: Vanilla | E-IRFS | GA-EIRFS.

Reuses the same inference/rendering building blocks as
find_and_render_comparison.py. The crop is ego-centered (fixed radius around
the ego vehicle at each frame) rather than world-frame-fixed, so it behaves
like a chase-cam view: objects naturally enter/exit frame as the car drives.

E-IRFS panel note: as elsewhere, this reuses the vanilla checkpoint's
predictions (see find_and_render_comparison.py's module docstring for why).

Usage:
    python scripts/render_comparison_video.py --center_idx 1324 \
        --out output_viz/scene_comparison.mp4 --fps 4
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
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENPCDET_ROOT = REPO_ROOT / 'OpenPCDet'
sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

TARGET_CLASSES = {'bicycle', 'motorcycle'}


def box_corners_2d(cx, cy, dx, dy, heading):
    hx, hy = dx / 2, dy / 2
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([cx, cy])


def run_inference(model, val_set, idx, score_thresh):
    data_dict = val_set.collate_batch([val_set[idx]])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
    keep = pred_scores >= score_thresh
    return pred_boxes[keep], pred_scores[keep], pred_labels[keep], data_dict


def find_scene_range(val_set, center_idx, gap_thresh=1.0):
    """Expand from center_idx while consecutive timestamps are <gap_thresh apart."""
    ts = lambda i: val_set.infos[i]['timestamp']
    lo, hi = center_idx, center_idx
    while lo - 1 >= 0 and ts(lo) - ts(lo - 1) < gap_thresh:
        lo -= 1
    while hi + 1 < len(val_set.infos) and ts(hi + 1) - ts(hi) < gap_thresh:
        hi += 1
    return lo, hi


def render_frame(idx, r, score_thresh, v_boxes, v_scores, v_labels,
                  g_boxes, g_scores, g_labels, points, gt, class_names):
    cx0, cy0 = 0.0, 0.0  # ego-centered crop

    def in_range(arr_xy):
        return (np.abs(arr_xy[:, 0] - cx0) <= r) & (np.abs(arr_xy[:, 1] - cy0) <= r)

    gt_c = gt[in_range(gt[:, :2])]
    pts_c = points[in_range(points[:, :2])]

    panels = [
        ('Vanilla', v_boxes, v_scores, v_labels),
        ('E-IRFS', v_boxes, v_scores, v_labels),
        ('GA-EIRFS (ours)', g_boxes, g_scores, g_labels),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=110)
    for ax, (title, pboxes, pscores, plabels) in zip(axes, panels):
        pmask = in_range(pboxes[:, :2]) if len(pboxes) else np.array([], dtype=bool)
        pb, pl = pboxes[pmask], plabels[pmask]

        ax.scatter(pts_c[:, 0], pts_c[:, 1], s=0.8, c=pts_c[:, 2], cmap='viridis', alpha=0.6, linewidths=0)
        for box in gt_c:
            gcx, gcy, dx, dy, heading, cls_idx = box[0], box[1], box[3], box[4], box[6], int(box[-1])
            corners = box_corners_2d(gcx, gcy, dx, dy, heading)
            name = class_names[cls_idx - 1] if 1 <= cls_idx <= len(class_names) else str(cls_idx)
            color = '#39ff6a' if name in TARGET_CLASSES else '#2fa84f'
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                                  linewidth=1.8 if name in TARGET_CLASSES else 0.8))

        for box, label in zip(pb, pl):
            pcx, pcy, dx, dy, heading = box[0], box[1], box[3], box[4], box[6]
            corners = box_corners_2d(pcx, pcy, dx, dy, heading)
            name = class_names[label - 1] if 1 <= label <= len(class_names) else str(label)
            color = '#ff3b3b' if name in TARGET_CLASSES else '#ff9d3b'
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                                  linewidth=1.8 if name in TARGET_CLASSES else 0.8, linestyle='--'))

        ax.plot(0, 0, marker='^', color='white', markersize=8)
        ax.set_xlim(cx0 - r, cx0 + r)
        ax.set_ylim(cy0 - r, cy0 + r)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'nuScenes val sample {idx} (green=GT, red/orange dashed=pred, score>={score_thresh})', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center_idx', type=int, required=True, help='any val index inside the scene to render')
    ap.add_argument('--score_thresh', type=float, default=0.3)
    ap.add_argument('--point_range', type=float, default=40.0)
    ap.add_argument('--fps', type=float, default=4.0, help='output video fps (2 native keyframes/sec, held longer for smoother playback)')
    ap.add_argument('--hold_frames', type=int, default=2, help='repeat each rendered frame this many times in the video')
    ap.add_argument('--out', type=str, default='output_viz/scene_comparison.mp4')
    args = ap.parse_args()

    import os
    os.chdir(OPENPCDET_ROOT / 'tools')

    vanilla_cfg_file = 'cfgs/nuscenes_models/centerpoint_trainval_vanilla.yaml'
    ga_cfg_file = 'cfgs/nuscenes_models/centerpoint_trainval_ga_eirfs.yaml'
    vanilla_ckpt = OPENPCDET_ROOT / 'output/nuscenes_models/centerpoint_trainval_vanilla/trainval20ep_seeded/ckpt/checkpoint_epoch_20.pth'
    ga_ckpt = OPENPCDET_ROOT / 'output/nuscenes_models/centerpoint_trainval_dias3d/trainval20ep_seeded/ckpt/checkpoint_epoch_20.pth'

    cfg_from_yaml_file(vanilla_cfg_file, cfg)
    logger = common_utils.create_logger()
    val_set, val_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, training=False, logger=logger,
    )
    class_names = cfg.CLASS_NAMES

    vanilla_model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=val_set)
    vanilla_model.load_params_from_file(filename=str(vanilla_ckpt), logger=logger, to_cpu=False)
    vanilla_model.cuda().eval()

    cfg_from_yaml_file(ga_cfg_file, cfg)
    ga_model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=val_set)
    ga_model.load_params_from_file(filename=str(ga_ckpt), logger=logger, to_cpu=False)
    ga_model.cuda().eval()

    lo, hi = find_scene_range(val_set, args.center_idx)
    print(f'Scene range: [{lo}, {hi}] ({hi - lo + 1} frames)')

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps, codec='libx264', quality=8)

    for idx in range(lo, hi + 1):
        v_boxes, v_scores, v_labels, data_dict = run_inference(vanilla_model, val_set, idx, args.score_thresh)
        g_boxes, g_scores, g_labels, _ = run_inference(ga_model, val_set, idx, args.score_thresh)
        points = data_dict['points'].cpu().numpy()[:, 1:4]
        gt = data_dict['gt_boxes'][0].cpu().numpy()
        gt = gt[np.any(gt != 0, axis=1)]

        frame = render_frame(idx, args.point_range, args.score_thresh, v_boxes, v_scores, v_labels,
                              g_boxes, g_scores, g_labels, points, gt, class_names)
        for _ in range(args.hold_frames):
            writer.append_data(frame)
        print(f'  rendered frame {idx} ({idx - lo + 1}/{hi - lo + 1})')

    writer.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
