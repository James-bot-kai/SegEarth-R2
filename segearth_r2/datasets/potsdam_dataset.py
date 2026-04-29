"""
Potsdam 4-channel (.npy) dataset adapter for SegEarth-R2.

Strips the NIR channel and converts Potsdam's JSON format to the dict
format expected by DataCollatorForCOCODatasetV2 / SegEarthR2.

Usage (eval):
    dataset = PotsdamDatasetForSegEarthR2(
        json_path='potsdam_test.json',
        base_dir='/data/potsdam',
        tokenizer=tokenizer,
        data_args=data_args,
        rgb_indices=[0, 1, 2],   # or [1, 2, 3] if channel order is IR-R-G-B
        split='test',
    )
    collator = DataCollatorForPotsdamDataset(
        tokenizer=tokenizer,
        clip_image_processor=clip_image_processor,
    )
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Sequence

import transformers
from tifffile import imread as tiff_imread

from segearth_r2.datasets.dataset import RS_Base_Dataset, DataCollatorForCOCODatasetV2, preprocess_mask
from segearth_r2.utils.constants import IGNORE_INDEX, REFER_TOKEN_INDEX


POTSDAM_CLASS_MAP = {
    'impervious surface': 0,
    'building':           1,
    'low vegetation':     2,
    'tree':               3,
    'car':                4,
    'background':         5,
}


class PotsdamDatasetForSegEarthR2(RS_Base_Dataset):
    """
    Reads Potsdam 4-channel .npy patches + LISA-style JSON,
    drops the NIR channel, and outputs the dict format that
    DataCollatorForCOCODatasetV2 / SegEarthR2 expect.
    """

    # ImageNet stats used by the Swin-B mask encoder
    PIXEL_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD  = torch.Tensor([58.395,  57.12,  57.375]).view(-1, 1, 1)

    def __init__(
        self,
        json_path: str,
        base_dir: str,
        tokenizer,
        data_args,
        rgb_indices: list = None,  # [0,1,2] for R-G-B-NIR; [1,2,3] for IR-R-G-B
        image_size: int = 1024,
        split: str = 'test',       # 'train' or 'test' — controls answer masking
    ):
        self.base_dir    = base_dir
        self.image_size  = image_size
        self.tokenizer   = tokenizer
        self.rgb_indices = rgb_indices if rgb_indices is not None else [0, 1, 2]
        self.split       = split
        self.SEG_token_id = self.tokenizer.convert_tokens_to_ids("[SEG]")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rgb(self, npy_path: str) -> np.ndarray:
        """Load .npy [H, W, 4], return uint8 [H, W, 3] RGB."""
        img = np.load(npy_path)                         # [H, W, 4]
        img_rgb = img[:, :, self.rgb_indices]           # [H, W, 3]
        if img_rgb.dtype != np.uint8:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.clip(0, 255).astype(np.uint8)
        return img_rgb

    def _resize_pad(self, arr: np.ndarray, target: int = 1024,
                    pad_val=128, interp=cv2.INTER_LINEAR) -> np.ndarray:
        """ResizeShortestEdge → FixedSizeCrop (same logic as preprocess_image)."""
        h, w = arr.shape[:2]
        scale = target / min(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        if max(new_h, new_w) > target:
            scale = target / max(new_h, new_w)
            new_h, new_w = int(new_h * scale + 0.5), int(new_w * scale + 0.5)
        resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)
        if arr.ndim == 2:
            padding = ((0, target - new_h), (0, target - new_w))
        else:
            padding = ((0, target - new_h), (0, target - new_w), (0, 0))
        return np.pad(resized, padding, mode='constant', constant_values=pad_val)

    def _preprocess_referring_instruction(self, instruction: str) -> torch.Tensor:
        tokenized = self.tokenizer.encode(instruction, add_special_tokens=False)
        seg_id    = self.tokenizer.encode('[SEG]', add_special_tokens=False)
        return torch.tensor(tokenized + seg_id)

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_path  = os.path.join(self.base_dir, item['image_path_rgb'])
        mask_path = os.path.join(self.base_dir, item['mask_path'])
        cls_str   = item['sampled_classes'][0]
        cls_idx   = POTSDAM_CLASS_MAP[cls_str.lower()]
        data_id   = item.get('id', idx)

        # ── 1. Load image (strip NIR) ──────────────────────────────────
        img_rgb = self._load_rgb(img_path)              # [H, W, 3] uint8

        # ── 2. Load mask ───────────────────────────────────────────────
        if mask_path.endswith('.npy'):
            mask_np = np.load(mask_path).astype(np.int64)
        else:
            mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        binary_mask = (mask_np == cls_idx).astype(np.uint8)  # [H, W]

        # ── 3. Preprocess image for Swin encoder ───────────────────────
        img_proc = self._resize_pad(img_rgb, self.image_size, pad_val=128)  # [1024,1024,3]
        image_tensor = torch.as_tensor(
            np.ascontiguousarray(img_proc.transpose(2, 0, 1))
        ).float()
        image_normalized = (image_tensor - self.PIXEL_MEAN) / self.PIXEL_STD  # [3,1024,1024]

        # ── 4. Preprocess mask (resize + pad, matching image transform) ─
        binary_mask_proc = self._resize_pad(
            binary_mask, self.image_size, pad_val=0, interp=cv2.INTER_NEAREST
        )  # [1024, 1024]

        # ── 5. Build conversation ──────────────────────────────────────
        #
        human_text = item['conversations'][0]['value']
        gpt_text   = item['conversations'][1]['value']
        # Strip image placeholder from description (goes into token_refer_id)
        description = (human_text
                       .replace('<image>', '')
                       .replace('<IMAGE>', '')
                       .strip())
        answer   = gpt_text          # must contain one or more [SEG] tokens
        mask_num = answer.count('[SEG]')

        token_refer_id = self._preprocess_referring_instruction(description)

        prefix = ('This is an image <|vision_bos|> <image> <|vision_eos|> '
                  '<|sep|> <|user|>, please doing Reasoning Segmentation '
                  'according to the following instruction:')
        sources = [[
            {'from': 'human', 'value': prefix + '\n<refer> <|assistant|>'},
            {'from': 'gpt',   'value': '\n' + answer},
        ]]
        text_dict = self.preprocess_llama2(sources, self.tokenizer)
        input_ids = text_dict['input_ids'][0]

        SEG_indices   = torch.zeros_like(input_ids)
        SEG_indices[input_ids == self.SEG_token_id] = 1

        refer_indices = torch.zeros_like(input_ids)
        refer_indices[input_ids == REFER_TOKEN_INDEX] = 1

        # ── 6. Build output dict ───────────────────────────────────────
        img_stem = os.path.basename(img_path).split('.')[0]

        data_dict = {
            'file_name':    img_path,     # kept for reference; collator overrides
            'image_rgb_raw': img_rgb,     # [H,W,3] uint8 — consumed by custom collator
            'height':       img_rgb.shape[0],
            'width':        img_rgb.shape[1],
            'image_id':     idx,
            'image':        image_normalized,
            'input_ids':    input_ids,
            'labels':       text_dict['labels'][0],
            'dataset_type': 'rs_reason_seg',
            'token_refer_id':             token_refer_id,
            'refer_embedding_indices':    refer_indices,
            'SEG_token_embedding_indices': SEG_indices,
            'mask_num':     mask_num,
            'annotations':  [],
        }

        for i in range(mask_num):
            data_dict['annotations'].append({
                'data_id':    data_id,
                'mask_id':    i,
                'mask':       np.expand_dims(binary_mask_proc, axis=0),  # [1,1024,1024]
                'image_path': img_path,
                'height':     img_rgb.shape[0],
                'width':      img_rgb.shape[1],
                'image_id':   img_stem,
            })

        return data_dict


@dataclass
class DataCollatorForPotsdamDataset(DataCollatorForCOCODatasetV2):
    """
    Subclasses the original collator, overriding images_clip construction
    so that it uses the in-memory RGB array from PotsdamDatasetForSegEarthR2
    instead of re-reading from disk with cv2.imread (which cannot open .npy).
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Pop raw RGB arrays before calling super (which would try cv2.imread)
        raw_rgb_list = [inst.pop('image_rgb_raw', None) for inst in instances]

        batch = super().__call__(instances)

        # Replace images_clip using our in-memory RGB uint8 arrays
        if raw_rgb_list and raw_rgb_list[0] is not None:
            image_clip = [
                self.clip_image_processor.preprocess(rgb, return_tensors='pt')['pixel_values'][0]
                for rgb in raw_rgb_list
            ]
            batch['images_clip'] = torch.stack(image_clip)

        return batch


# ──────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-6))


def evaluate_predictions(
    pred_dir: str,
    json_path: str,
    base_dir: str,
    split_name: str = 'test_data',   # matches the tif filename pattern
    rgb_indices: list = None,
    verbose: bool = True,
):
    """
    Compare SegEarth-R2 .tif predictions against Potsdam GT masks.

    Prediction filename pattern (set by eval.py):
        {image_stem}_{data_id}_{split_name}_{mask_id}.tif

    Args:
        pred_dir:   directory containing .tif output files
        json_path:  path to your LISA-style test JSON
        base_dir:   root dir for mask_path entries in the JSON
        split_name: the split token in the tif filename (e.g. 'test_data')
    """
    with open(json_path, encoding='utf-8') as f:
        samples = json.load(f)

    ious, dices = [], []
    missing = []

    for item in samples:
        img_stem  = os.path.basename(item['image_path_rgb']).split('.')[0]
        mask_path = os.path.join(base_dir, item['mask_path'])
        cls_str   = item['sampled_classes'][0]
        cls_idx   = POTSDAM_CLASS_MAP[cls_str.lower()]
        data_id   = item.get('id', 0)
        mask_num  = item['conversations'][1]['value'].count('[SEG]')

        # Load GT binary mask
        if mask_path.endswith('.npy'):
            mask_np = np.load(mask_path).astype(np.int64)
        else:
            mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        gt = (mask_np == cls_idx)

        for mask_id in range(mask_num):
            pred_name = f"{img_stem}_{data_id}_{split_name}_{mask_id}.tif"
            pred_path = os.path.join(pred_dir, pred_name)

            if not os.path.exists(pred_path):
                missing.append(pred_name)
                continue

            pred_raw = tiff_imread(pred_path)   # uint8, 0 or 255
            pred = pred_raw > 0

            # Resize pred back to GT resolution if padded/resized
            if pred.shape != gt.shape:
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            ious.append(compute_iou(pred, gt))
            dices.append(compute_dice(pred, gt))

    if verbose:
        print(f"Evaluated {len(ious)} instances  |  missing: {len(missing)}")
        print(f"  mIoU  = {np.mean(ious):.4f}")
        print(f"  mDice = {np.mean(dices):.4f}")
        if missing:
            print(f"  Missing files (first 5): {missing[:5]}")

    return {
        'mIoU':   float(np.mean(ious))  if ious  else 0.0,
        'mDice':  float(np.mean(dices)) if dices else 0.0,
        'n':      len(ious),
        'missing': missing,
    }


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json',       required=True)
    parser.add_argument('--base_dir',   required=True)
    parser.add_argument('--pred_dir',   default=None, help='If set, run evaluation')
    parser.add_argument('--split_name', default='test_data')
    args = parser.parse_args()

    if args.pred_dir:
        results = evaluate_predictions(
            pred_dir=args.pred_dir,
            json_path=args.json,
            base_dir=args.base_dir,
            split_name=args.split_name,
        )
    else:
        print("Pass --pred_dir to run evaluation.")
