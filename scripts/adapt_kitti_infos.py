"""
Translate OpenPCDet's native KITTI infos pkl into the flat field format our
GA-EIRFS sampler code already expects (built originally for nuScenes' infos
format).

Why this exists: our sampler/geometry_score code reads info['gt_names'],
info['num_lidar_pts'], info['gt_boxes'], info['lidar_path'] at the TOP level
of each frame's info dict (nuScenes' native format). KITTI's native infos
nest the equivalent fields under info['annos'] with different names
(info['annos']['name'], info['annos']['num_points_in_gt'],
info['annos']['gt_boxes_lidar']) and identify the point cloud file by a
numeric info['point_cloud']['lidar_idx'] rather than a path string. Rather
than touch the already-validated nuScenes code paths, this script produces a
second pkl with the extra top-level fields added (originals kept intact),
so the existing sampler works unchanged against either dataset.

Box format note: KITTI's gt_boxes_lidar is already [x,y,z,l,w,h,heading] --
this IS OpenPCDet's unified box convention (dx,dy,dz), same axis meaning as
nuScenes' gt_boxes[:7], so no reordering is needed, only field renaming.

DontCare handling: annos['name'] includes trailing 'DontCare' placeholder
entries (by KITTI convention, always listed after the real objects), which
annos['gt_boxes_lidar']/['num_points_in_gt'] already exclude (they're sized
to num_objects, not num_gt). We truncate 'name' the same way so gt_names
lines up 1:1 with gt_boxes.

Class filtering: the standard KITTI 3D benchmark (and our PointPillars
configs) only train on Car/Pedestrian/Cyclist, but raw KITTI labels include
5 more classes (Van, Truck, Tram, Misc, Person_sitting). Unlike nuScenes
(where all 10 label classes exactly match CLASS_NAMES, so the sampler's
frequency/G_c code never needed a filter), here we filter gt_names/gt_boxes/
num_lidar_pts down to --classes at adapter time -- keeping the core sampler
code (which has no class-filtering logic) unchanged, same "adapter layer,
don't touch validated code" philosophy as the rest of this script.

Usage:
    python scripts/adapt_kitti_infos.py \
        --data_root OpenPCDet/data/kitti \
        --split train
    python scripts/adapt_kitti_infos.py --data_root OpenPCDet/data/kitti --split val
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def adapt_infos(infos, split, classes=None):
    adapted = []
    n_dropped_mismatch = 0
    for info in infos:
        annos = info.get('annos')
        if annos is None:
            # has_label=False (e.g. test split with no ground truth) -- nothing to adapt
            adapted.append(info)
            continue

        gt_boxes = annos['gt_boxes_lidar']
        num_objects = len(gt_boxes)
        names = annos['name'][:num_objects]
        num_pts = annos['num_points_in_gt'][:num_objects]

        if len(names) != num_objects or len(num_pts) != num_objects:
            # defensive: should not happen given the KITTI convention this
            # relies on (DontCare always trailing), but don't silently
            # misalign data if it does
            n_dropped_mismatch += 1
            continue

        if classes is not None:
            keep = np.array([n in classes for n in names])
            names = np.array(names)[keep]
            num_pts = np.array(num_pts)[keep]
            gt_boxes = gt_boxes[keep]

        lidar_idx = info['point_cloud']['lidar_idx']
        # train/val both live under the 'training' split dir in raw KITTI
        # (only the held-out 'testing' set, which has no labels, differs)
        lidar_path = f"training/velodyne/{lidar_idx}.bin"

        info['gt_names'] = names
        info['num_lidar_pts'] = num_pts
        info['gt_boxes'] = gt_boxes
        info['lidar_path'] = lidar_path
        info['num_point_features'] = info['point_cloud'].get('num_features', 4)
        adapted.append(info)

    if n_dropped_mismatch:
        print(f'WARNING: {n_dropped_mismatch}/{len(infos)} frames had a '
              f'name/box count mismatch (unexpected DontCare ordering?) and '
              f'were left without adapted fields -- inspect before trusting '
              f'downstream G_c/frequency numbers.')
    return adapted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='OpenPCDet/data/kitti')
    ap.add_argument('--split', type=str, required=True, choices=['train', 'val'])
    ap.add_argument('--classes', type=str, nargs='*', default=['Car', 'Pedestrian', 'Cyclist'],
                     help='Classes to keep (matches the standard KITTI 3D benchmark / our '
                          'PointPillars CLASS_NAMES). Pass --classes with no values to disable filtering.')
    args = ap.parse_args()
    classes = set(args.classes) if args.classes else None

    data_root = Path(args.data_root)
    in_path = data_root / f'kitti_infos_{args.split}.pkl'
    out_path = data_root / f'kitti_infos_{args.split}_adapted.pkl'

    print(f'Loading {in_path}')
    with open(in_path, 'rb') as f:
        infos = pickle.load(f)
    print(f'Loaded {len(infos)} frames')
    print(f'Filtering to classes: {sorted(classes) if classes else "(none -- keeping all)"}')

    adapted = adapt_infos(infos, args.split, classes=classes)

    # quick sanity print
    from collections import Counter
    class_counts = Counter()
    frame_counts = Counter()
    total_frames_with_annos = 0
    for info in adapted:
        if 'gt_names' not in info:
            continue
        total_frames_with_annos += 1
        seen = set()
        for name in info['gt_names']:
            class_counts[name] += 1
            seen.add(name)
        for name in seen:
            frame_counts[name] += 1
    print(f'\nFrames with annotations: {total_frames_with_annos}/{len(adapted)}')
    print(f"{'class':<15}{'instances':>12}{'frames':>10}")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c]):
        print(f"{cls:<15}{class_counts[cls]:>12}{frame_counts[cls]:>10}")

    with open(out_path, 'wb') as f:
        pickle.dump(adapted, f)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
