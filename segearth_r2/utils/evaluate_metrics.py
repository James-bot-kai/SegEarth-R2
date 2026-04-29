"""
Evaluation metrics for SegEarth-R2 on Potsdam dataset.

Metrics:
  gIoU  - Global IoU:  sum(all intersections) / sum(all unions)
  cIoU  - Class IoU:   per-class mean IoU
  mIoU  - Mean IoU:    mean of per-sample IoU
  mAcc  - Mean Acc:    per-class mean pixel accuracy (TP / (TP+FN))
  mF1   - Mean F1:     mean of per-sample Dice/F1

Usage:
    python segearth_r2/utils/evaluate_metrics.py \
        --pred_dir   results/potsdam_eval \
        --json_path  /root/SegImage_Output/step2_dataset_qwen_single_turn.json \
        --base_dir   /root/SegImage_Output \
        --split_name test_data
"""

import os
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from tifffile import imread as tiff_imread

POTSDAM_CLASS_MAP = {
    'impervious surface': 0,
    'building':           1,
    'low vegetation':     2,
    'tree':               3,
    'car':                4,
    'background':         5,
}
CLASS_NAMES = {v: k for k, v in POTSDAM_CLASS_MAP.items()}


def load_gt_binary(mask_path: str, cls_idx: int, pred_shape=None):
    """Load GT mask and extract binary mask for the given class."""
    if mask_path.endswith('.npy'):
        mask_np = np.load(mask_path).astype(np.int64)
    else:
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
    gt = (mask_np == cls_idx).astype(bool)

    # Resize GT to match prediction if needed
    if pred_shape is not None and gt.shape != pred_shape:
        gt = cv2.resize(gt.astype(np.uint8),
                        (pred_shape[1], pred_shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return gt


def compute_sample_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Compute per-sample metrics for a binary prediction/GT pair.
    Returns dict with: iou, f1, acc, tp, fp, fn, tn
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    tp = np.logical_and(pred,  gt).sum()
    fp = np.logical_and(pred,  ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    iou  = tp / (tp + fp + fn + 1e-6)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-6)
    acc  = tp / (tp + fn + 1e-6)   # recall / sensitivity (foreground accuracy)

    return dict(iou=float(iou), f1=float(f1), acc=float(acc),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def evaluate_all(pred_dir, json_path, base_dir, split_name='test_data', verbose=True):
    """
    Compute gIoU, cIoU, mIoU, mAcc, mF1 for SegEarth-R2 predictions on Potsdam.

    Returns a dict with all metrics.
    """
    with open(json_path, encoding='utf-8') as f:
        samples = json.load(f)

    # Per-sample accumulators
    all_ious, all_f1s, all_accs = [], [], []

    # Per-class accumulators for cIoU and mAcc
    class_tp   = defaultdict(int)
    class_fp   = defaultdict(int)
    class_fn   = defaultdict(int)
    class_cnt  = defaultdict(int)
    class_ious = defaultdict(list)
    class_accs = defaultdict(list)

    # Global accumulators for gIoU
    global_intersection = 0
    global_union        = 0

    missing = []

    for uid, item in enumerate(samples):
        img_stem  = os.path.basename(item['image_path_rgb']).replace('.npy', '')
        mask_path = os.path.join(base_dir, item['mask_path'])
        cls_str   = item['sampled_classes'][0].lower()
        cls_idx   = POTSDAM_CLASS_MAP.get(cls_str, -1)
        data_id   = uid  # matches how convert script sets id=uid

        n_seg = item['conversations'][1]['value'].count('[SEG]')

        for mask_id in range(max(n_seg, 1)):
            pred_name = f"{img_stem}_{data_id}_{split_name}_{mask_id}.tif"
            pred_path = os.path.join(pred_dir, pred_name)

            if not os.path.exists(pred_path):
                missing.append(pred_name)
                continue

            pred_raw = tiff_imread(pred_path)
            pred = (pred_raw > 0).astype(bool)

            gt = load_gt_binary(mask_path, cls_idx, pred_shape=pred.shape)

            m = compute_sample_metrics(pred, gt)

            # Per-sample metrics
            all_ious.append(m['iou'])
            all_f1s.append(m['f1'])
            all_accs.append(m['acc'])

            # Per-class metrics
            class_tp[cls_idx]  += m['tp']
            class_fp[cls_idx]  += m['fp']
            class_fn[cls_idx]  += m['fn']
            class_cnt[cls_idx] += 1
            class_ious[cls_idx].append(m['iou'])
            class_accs[cls_idx].append(m['acc'])

            # Global gIoU accumulators
            global_intersection += m['tp']
            global_union        += m['tp'] + m['fp'] + m['fn']

    # ── Compute final metrics ──────────────────────────────────────────
    gIoU = global_intersection / (global_union + 1e-6)
    mIoU = float(np.mean(all_ious)) if all_ious else 0.0
    mF1  = float(np.mean(all_f1s))  if all_f1s  else 0.0
    mAcc_sample = float(np.mean(all_accs)) if all_accs else 0.0

    # cIoU: per-class IoU (aggregate TP/FP/FN per class, then average)
    class_iou_vals = {}
    class_acc_vals = {}
    for cid in sorted(class_tp.keys()):
        tp_ = class_tp[cid]
        fp_ = class_fp[cid]
        fn_ = class_fn[cid]
        class_iou_vals[cid] = tp_ / (tp_ + fp_ + fn_ + 1e-6)
        class_acc_vals[cid] = tp_ / (tp_ + fn_ + 1e-6)

    cIoU = float(np.mean(list(class_iou_vals.values()))) if class_iou_vals else 0.0
    mAcc = float(np.mean(list(class_acc_vals.values()))) if class_acc_vals else 0.0

    # ── Print results ──────────────────────────────────────────────────
    if verbose:
        print("\n" + "="*55)
        print(f"  Evaluated : {len(all_ious)} instances  |  missing: {len(missing)}")
        print("="*55)
        print(f"  gIoU  = {gIoU:.4f}   (global intersection/union)")
        print(f"  cIoU  = {cIoU:.4f}   (per-class aggregate IoU mean)")
        print(f"  mIoU  = {mIoU:.4f}   (per-sample IoU mean)")
        print(f"  mAcc  = {mAcc:.4f}   (per-class recall mean)")
        print(f"  mF1   = {mF1:.4f}   (per-sample Dice/F1 mean)")
        print("="*55)

        print("\nPer-class breakdown:")
        print(f"  {'Class':<22} {'IoU':>6}  {'Acc':>6}  {'Count':>5}")
        print("  " + "-"*44)
        for cid in sorted(class_iou_vals.keys()):
            name = CLASS_NAMES.get(cid, f"cls_{cid}")
            print(f"  {name:<22} {class_iou_vals[cid]:>6.4f}  "
                  f"{class_acc_vals[cid]:>6.4f}  {class_cnt[cid]:>5}")

        if missing:
            print(f"\nMissing files (first 5): {missing[:5]}")

    return {
        'gIoU': gIoU, 'cIoU': cIoU, 'mIoU': mIoU,
        'mAcc': mAcc, 'mF1':  mF1,
        'per_class_iou': class_iou_vals,
        'per_class_acc': class_acc_vals,
        'n': len(all_ious), 'missing': missing,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',   required=True)
    parser.add_argument('--json_path',  required=True)
    parser.add_argument('--base_dir',   required=True)
    parser.add_argument('--split_name', default='test_data')
    args = parser.parse_args()

    evaluate_all(
        pred_dir=args.pred_dir,
        json_path=args.json_path,
        base_dir=args.base_dir,
        split_name=args.split_name,
    )
