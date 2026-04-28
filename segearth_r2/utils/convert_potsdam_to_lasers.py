"""
Convert LISA-style Potsdam JSON → LaSeRS-compatible JSON for SegEarth-R2.

The key differences between the two formats:
  LISA JSON  :  image_path_rgb (→ .npy), mask_path, sampled_classes, conversations
  LaSeRS JSON:  image_name (→ .png), description, answer, id [, mask (RLE list)]

This script also:
  1. Strips the NIR channel from each .npy file and saves RGB .png images.
  2. Converts binary masks to COCO RLE format (optional; skip if no GT needed).

Usage:
    python tools/convert_potsdam_to_lasers.py \
        --lisa_json   /data/potsdam/test.json \
        --base_dir    /data/potsdam \
        --out_img_dir /data/potsdam_rgb/test/images \
        --out_ann     /data/potsdam_rgb/test/annotations/test_data.json \
        --rgb_indices 0 1 2
    # 转换测试集（含 GT mask，用于后续评估指标计算）
    python segearth_r2/utils/convert_potsdam_to_lasers.py \
    --lisa_json /root/SegImage_Output/step2_dataset_qwen_single_turn.json \
    --base_dir /root/SegImage_Output \
    --out_img_dir /root/SegImage_Output/images_rgb_vis \
    --out_ann /root/SegImage_Output/test_data.json \
    --rgb_indices 0 1 2
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as cocomask

POTSDAM_CLASS_MAP = {
    'impervious surface': 0,
    'building':           1,
    'low vegetation':     2,
    'tree':               3,
    'car':                4,
    'background':         5,
}


def load_npy_rgb(npy_path: str, rgb_indices=(0, 1, 2)) -> np.ndarray:
    """Load [H,W,4] .npy → uint8 [H,W,3] RGB."""
    img = np.load(npy_path)
    img_rgb = img[:, :, list(rgb_indices)]
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8) if img_rgb.max() <= 1.0 \
                  else img_rgb.clip(0, 255).astype(np.uint8)
    return img_rgb


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a uint8 binary mask [H, W] to COCO RLE dict."""
    mask_f = np.asfortranarray(mask.astype(np.uint8))
    rle = cocomask.encode(mask_f)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def convert(lisa_json, base_dir, out_img_dir, out_ann, rgb_indices, include_mask=True):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)

    with open(lisa_json, encoding='utf-8') as f:
        samples = json.load(f)

    output = []
    for uid, item in enumerate(samples):
        npy_path  = os.path.join(base_dir, item['image_path_rgb'])
        mask_path = os.path.join(base_dir, item['mask_path'])
        cls_str   = item['sampled_classes'][0]
        cls_idx   = POTSDAM_CLASS_MAP[cls_str.lower()]

        # -- Save RGB image --------------------------------------------------
        img_rgb = load_npy_rgb(npy_path, rgb_indices)
        img_name = os.path.basename(npy_path).replace('.npy', '.png')
        out_path = os.path.join(out_img_dir, img_name)
        if not os.path.exists(out_path):
            Image.fromarray(img_rgb).save(out_path)

        # -- Build binary mask in RLE ----------------------------------------
        rle_list = None
        if include_mask:
            if mask_path.endswith('.npy'):
                mask_np = np.load(mask_path).astype(np.int64)
            else:
                mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
            binary = (mask_np == cls_idx).astype(np.uint8)
            # One RLE per [SEG] token in the answer
            n_seg = item['conversations'][1]['value'].count('[SEG]')
            rle_single = binary_mask_to_rle(binary)
            rle_list = [rle_single] * max(n_seg, 1)  # 每个 [SEG] 对应一个 mask

        # -- Build LaSeRS entry ----------------------------------------------
        description = (item['conversations'][0]['value']
                       .replace('<image>', '').replace('<IMAGE>', '').strip())
        answer = item['conversations'][1]['value']

        entry = {
            'id':          uid,           # 必须是数字，原始字符串 id 含 / 会导致文件名非法
            'image_name':  img_name,
            'description': description,
            'answer':      answer,
        }
        if rle_list is not None:
            entry['mask'] = rle_list

        output.append(entry)
        if (uid + 1) % 100 == 0:
            print(f"  converted {uid+1}/{len(samples)}")

    with open(out_ann, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(output)} entries saved to {out_ann}")
    print(f"Images saved to {out_img_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lisa_json',   required=True,  help='LISA-style input JSON')
    parser.add_argument('--base_dir',    required=True,  help='Root dir for npy/mask paths')
    parser.add_argument('--out_img_dir', required=True,  help='Output dir for RGB .png images')
    parser.add_argument('--out_ann',     required=True,  help='Output annotation JSON path')
    parser.add_argument('--rgb_indices', nargs=3, type=int, default=[0, 1, 2],
                        metavar=('R', 'G', 'B'),
                        help='Channel indices for RGB in the .npy (e.g. 1 2 3 for IR-R-G-B)')
    parser.add_argument('--no_mask', action='store_true',
                        help='Skip GT mask conversion (for pure inference)')
    args = parser.parse_args()

    convert(
        lisa_json=args.lisa_json,
        base_dir=args.base_dir,
        out_img_dir=args.out_img_dir,
        out_ann=args.out_ann,
        rgb_indices=tuple(args.rgb_indices),
        include_mask=not args.no_mask,
    )
