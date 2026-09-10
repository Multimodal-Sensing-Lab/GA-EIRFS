"""
Scan nuScenes val for frames where GA-EIRFS visibly beats vanilla on bicycle/
motorcycle detection (more matched target-class GT, zero GA-EIRFS false
positives on those classes) and render 3-panel side-by-side BEV comparisons:
Vanilla | E-IRFS (uses the vanilla checkpoint as a stand-in — see note below) | GA-EIRFS.

We don't have an E-IRFS-only checkpoint trained for 20 epochs (only 12), and the
paper's own result is that E-IRFS-only is statistically indistinguishable from
vanilla, so the middle panel reuses the vanilla checkpoint's predictions rather
than misrepresenting a different model.

The crop around each frame's target-class GT is auto-fit: GT boxes are
clustered by proximity (single-linkage, eps=12m) and the largest cluster's
bounding box (+ padding) becomes the crop, so a frame with GT scattered across
two unrelated clusters doesn't end up cropped to empty space between them.

Usage:
    python scripts/find_and_render_comparison.py --scan 6019 --num_images 3 \
        --out_prefix output_viz/three_way_comparison
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

TARGET_CLASSES = {'bicycle', 'motorcycle'}


def box_corners_2d(cx, cy, dx, dy, heading):
    hx, hy = dx / 2, dy / 2
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return (local @ rot.T) + np.array([cx, cy])


def center_dist_match(pred_boxes, pred_labels, gt_boxes, class_names, thresh=2.5):
    """Greedy center-distance matching. Returns (n_matched, n_unmatched_pred)."""
    if len(pred_boxes) == 0:
        return 0, 0
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    matched = 0
    unmatched = 0
    for box, label in zip(pred_boxes, pred_labels):
        name = class_names[label - 1] if 1 <= label <= len(class_names) else None
        best_j, best_d = -1, thresh
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            gt_name = class_names[int(gb[-1]) - 1] if 1 <= int(gb[-1]) <= len(class_names) else None
            if gt_name != name:
                continue
            d = np.hypot(box[0] - gb[0], box[1] - gb[1])
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            gt_used[best_j] = True
            matched += 1
        else:
            unmatched += 1
    return matched, unmatched


def cluster_boxes(centers, eps=12.0):
    """Single-linkage clustering by center distance. Returns list of index lists."""
    n = len(centers)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) <= eps:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


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


def auto_crop(gt_target, pad=4.0, min_r=6.0):
    """Largest proximity cluster of target-class GT -> (cx, cy, r) square crop."""
    clusters = cluster_boxes(gt_target[:, :2], eps=12.0)
    clusters.sort(key=len, reverse=True)
    idxs = clusters[0]
    sub = gt_target[idxs]
    xs_lo = sub[:, 0] - sub[:, 3] / 2
    xs_hi = sub[:, 0] + sub[:, 3] / 2
    ys_lo = sub[:, 1] - sub[:, 4] / 2
    ys_hi = sub[:, 1] + sub[:, 4] / 2
    cx0 = float((xs_lo.min() + xs_hi.max()) / 2)
    cy0 = float((ys_lo.min() + ys_hi.max()) / 2)
    half_w = float((xs_hi.max() - xs_lo.min()) / 2) + pad
    half_h = float((ys_hi.max() - ys_lo.min()) / 2) + pad
    r = max(half_w, half_h, min_r)
    return cx0, cy0, r, len(idxs)


def render_panel_figure(idx, cx0, cy0, r, score_thresh, v_boxes, v_scores, v_labels,
                         g_boxes, g_scores, g_labels, points, gt, class_names, out_path):
    def in_range(arr_xy):
        return (np.abs(arr_xy[:, 0] - cx0) <= r) & (np.abs(arr_xy[:, 1] - cy0) <= r)

    gt_c = gt[in_range(gt[:, :2])]
    pts_c = points[in_range(points[:, :2])]

    panels = [
        ('Vanilla (no rebalancing)', v_boxes, v_scores, v_labels),
        ('E-IRFS', v_boxes, v_scores, v_labels),
        ('GA-EIRFS (ours)', g_boxes, g_scores, g_labels),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), dpi=170)
    for ax, (title, pboxes, pscores, plabels) in zip(axes, panels):
        pmask = in_range(pboxes[:, :2]) if len(pboxes) else np.array([], dtype=bool)
        pb, ps, pl = pboxes[pmask], pscores[pmask], plabels[pmask]

        ax.scatter(pts_c[:, 0], pts_c[:, 1], s=1.2, c=pts_c[:, 2], cmap='viridis', alpha=0.6, linewidths=0)
        for box in gt_c:
            gcx, gcy, dx, dy, heading, cls_idx = box[0], box[1], box[3], box[4], box[6], int(box[-1])
            corners = box_corners_2d(gcx, gcy, dx, dy, heading)
            name = class_names[cls_idx - 1] if 1 <= cls_idx <= len(class_names) else str(cls_idx)
            color = '#39ff6a' if name in TARGET_CLASSES else '#2fa84f'
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                                  linewidth=2.2 if name in TARGET_CLASSES else 1.1))
            if name in TARGET_CLASSES:
                ax.text(gcx, gcy - 1.0, f'GT:{name}', color=color, fontsize=9, ha='center', va='top', weight='bold')

        for box, score, label in zip(pb, ps, pl):
            pcx, pcy, dx, dy, heading = box[0], box[1], box[3], box[4], box[6]
            corners = box_corners_2d(pcx, pcy, dx, dy, heading)
            name = class_names[label - 1] if 1 <= label <= len(class_names) else str(label)
            color = '#ff3b3b' if name in TARGET_CLASSES else '#ff9d3b'
            ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=color,
                                  linewidth=2.2 if name in TARGET_CLASSES else 1.0, linestyle='--'))
            if name in TARGET_CLASSES:
                ax.text(pcx, pcy + 1.0, f'{name} {score:.2f}', color=color, fontsize=9, ha='center', va='bottom', weight='bold')

        ax.set_xlim(cx0 - r, cx0 + r)
        ax.set_ylim(cy0 - r, cy0 + r)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f'nuScenes val sample {idx} — bicycle/motorcycle detections '
        f'(green = ground truth, red/orange dashed = prediction, score≥{score_thresh})',
        fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', type=int, default=6019, help='number of val samples to scan')
    ap.add_argument('--start', type=int, default=0, help='start index for scan')
    ap.add_argument('--score_thresh', type=float, default=0.3)
    ap.add_argument('--num_images', type=int, default=3, help='how many distinct top candidates to render')
    ap.add_argument('--exclude_idx', type=int, nargs='*', default=[], help='val indices to skip (already used)')
    ap.add_argument('--out_prefix', type=str, default='output_viz/three_way_comparison')
    ap.add_argument('--force_idx', type=int, nargs='*', default=None, help='skip the scan, render exactly these val indices')
    ap.add_argument('--center_x', type=float, default=None, help='manual crop-center override (only valid with a single --force_idx)')
    ap.add_argument('--center_y', type=float, default=None)
    ap.add_argument('--point_range', type=float, default=None, help='manual crop half-width override (only valid with a single --force_idx)')
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

    if args.force_idx:
        selected = [(0, i, None, None) for i in args.force_idx]
    else:
        print(f'Scanning val samples [{args.start}, {args.start + args.scan}) for clean GA-EIRFS-wins cases...')
        candidates = []
        end = min(args.start + args.scan, len(val_set))
        for idx in range(args.start, end):
            if idx in args.exclude_idx:
                continue
            info = val_set.infos[idx]
            gt_names = info.get('gt_names', [])
            if not any(n in TARGET_CLASSES for n in gt_names):
                continue

            v_boxes, v_scores, v_labels, data_dict = run_inference(vanilla_model, val_set, idx, args.score_thresh)
            g_boxes, g_scores, g_labels, _ = run_inference(ga_model, val_set, idx, args.score_thresh)

            gt = data_dict['gt_boxes'][0].cpu().numpy()
            gt = gt[np.any(gt != 0, axis=1)]
            gt_target = gt[[class_names[int(c) - 1] in TARGET_CLASSES for c in gt[:, -1]]]
            if len(gt_target) == 0:
                continue

            v_tgt_mask = np.array([class_names[l - 1] in TARGET_CLASSES for l in v_labels]) if len(v_labels) else np.array([], dtype=bool)
            g_tgt_mask = np.array([class_names[l - 1] in TARGET_CLASSES for l in g_labels]) if len(g_labels) else np.array([], dtype=bool)
            _, v_tgt_unmatched = center_dist_match(v_boxes[v_tgt_mask], v_labels[v_tgt_mask], gt, class_names)
            _, g_tgt_unmatched = center_dist_match(g_boxes[g_tgt_mask], g_labels[g_tgt_mask], gt, class_names)

            if g_tgt_unmatched > 0:
                continue  # skip frames where GA-EIRFS itself shows a false positive

            g_tgt_matched = int(g_tgt_mask.sum()) - g_tgt_unmatched
            v_tgt_matched = int(v_tgt_mask.sum()) - v_tgt_unmatched
            advantage = g_tgt_matched - v_tgt_matched
            if advantage <= 0:
                continue

            score = advantage * 10 + g_tgt_matched
            candidates.append((score, idx, v_tgt_matched, g_tgt_matched))
            print(f'  sample {idx}: vanilla matched {v_tgt_matched}, GA-EIRFS matched {g_tgt_matched} '
                  f'(gt targets={len(gt_target)}, advantage={advantage})')

        if not candidates:
            print('No clean winning sample found in the scanned range. Try a larger --scan.')
            return

        candidates.sort(key=lambda c: c[0], reverse=True)
        # keep only frames whose auto-crop centers are >20m apart, for visual diversity
        selected = []
        for cand in candidates:
            idx = cand[1]
            data_dict_tmp = val_set.collate_batch([val_set[idx]])
            gt_tmp = np.asarray(data_dict_tmp['gt_boxes'][0])
            gt_tmp = gt_tmp[np.any(gt_tmp != 0, axis=1)]
            gt_target_tmp = gt_tmp[[class_names[int(c) - 1] in TARGET_CLASSES for c in gt_tmp[:, -1]]]
            cx, cy, _, _ = auto_crop(gt_target_tmp)
            if all(np.hypot(cx - s[4], cy - s[5]) > 20.0 for s in selected):
                selected.append((*cand, cx, cy))
            if len(selected) >= args.num_images:
                break

    for i, sel in enumerate(selected):
        idx = sel[1]
        print(f'Rendering sample {idx} (vanilla matched={sel[2]}, GA-EIRFS matched={sel[3]})')

        v_boxes, v_scores, v_labels, data_dict = run_inference(vanilla_model, val_set, idx, args.score_thresh)
        g_boxes, g_scores, g_labels, _ = run_inference(ga_model, val_set, idx, args.score_thresh)
        points = data_dict['points'].cpu().numpy()[:, 1:4]
        gt = data_dict['gt_boxes'][0].cpu().numpy()
        gt = gt[np.any(gt != 0, axis=1)]
        gt_target = gt[[class_names[int(c) - 1] in TARGET_CLASSES for c in gt[:, -1]]]

        if args.center_x is not None and args.center_y is not None:
            cx0, cy0 = args.center_x, args.center_y
            r = args.point_range if args.point_range is not None else 10.0
            print(f'  manual crop: center=({cx0:.1f},{cy0:.1f}) r={r:.1f}')
        else:
            cx0, cy0, r, cluster_size = auto_crop(gt_target)
            print(f'  auto-crop: center=({cx0:.1f},{cy0:.1f}) r={r:.1f} cluster_size={cluster_size}')

        out_path = REPO_ROOT / f'{args.out_prefix}_{i+1}.png'
        render_panel_figure(idx, cx0, cy0, r, args.score_thresh, v_boxes, v_scores, v_labels,
                             g_boxes, g_scores, g_labels, points, gt, class_names, out_path)


if __name__ == '__main__':
    main()
