#!/usr/bin/env python3
"""
AerialVG Official-Protocol Evaluation: Top-1 / Top-5 Accuracy @ IoU > 0.5.

Follows the protocol in the official AerialVG eval.py
(https://github.com/Ideal-ljl/AerialVG/blob/main/eval.py):

  - For each sample, use only the first GT box (boxes[0]) — single-target
  - Per-query confidence: pred_logits.sigmoid().max(dim=text_tokens)
  - Top-K (K=5) queries selected by confidence
  - Top-1 Acc: 1 if IoU(top1_pred, gt) > 0.5
  - Top-5 Acc: 1 if max IoU(top5_preds, gt) > 0.5
  - Threshold uses strict > (not >=)

Results comparable to AerialVG paper Table 1.

Usage:
    # CCM model (default)
    CUDA_VISIBLE_DEVICES=0 python eval_topk_acc.py \
        --checkpoint outputs/output_CCM/latest.pth --split test

    # Base model (disable CCM since untrained)
    CUDA_VISIBLE_DEVICES=0 python eval_topk_acc.py \
        --checkpoint outputs/train_base/latest.pth --split test --no_ccm
"""
import argparse
import json
import os
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_iou


def parse_args():
    parser = argparse.ArgumentParser("AerialVG official Top-K Accuracy evaluation")
    parser.add_argument("--config_file",
                        default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="/data2/huggingface/AerialVG")
    parser.add_argument("--image_dir", default="/data2/huggingface/AerialVG/images")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--no_ccm", action="store_true",
                        help="Disable CCM (use fixed 900 queries). For Base model.")
    return parser.parse_args()


def preprocess_image(image, size=800):
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    return tensor


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = device
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)), strict=False)

    # Configure CCM/GPQ behavior at inference
    if args.no_ccm:
        model.transformer.use_ccm = False
        print("  CCM disabled (fixed 900 queries)")
    else:
        model.transformer.use_ccm = True
        print(f"  CCM enabled, dynamic_query_list = {model.transformer.dynamic_query_list}")

    # Auto-detect GPQ: if active_mask has any False, this is a GPQ checkpoint
    if (~model.transformer.query_active_mask).any().item():
        model.transformer.use_gpq = True
        n_active = int(model.transformer.query_active_mask.sum().item())
        n_total = model.transformer.query_active_mask.numel()
        print(f"  GPQ detected: {n_active}/{n_total} active slots")

    model.eval().to(device)

    # Load dataset
    print(f"Loading dataset from {args.data_dir} (split={args.split})...")
    raw_ds = load_dataset(args.data_dir)
    if args.split not in raw_ds:
        args.split = list(raw_ds.keys())[-1]
    test_data = raw_ds[args.split]
    print(f"  {len(test_data)} samples")

    n_total = 0
    n_top1 = 0
    n_topk = 0
    first_iou_sum = 0.0
    max_iou_sum = 0.0

    for idx in tqdm(range(len(test_data)), desc="Evaluating"):
        item = test_data[idx]
        filename = item["filename"]
        orig_h = item["height"]
        orig_w = item["width"]
        regions = item["grounding"]["regions"]
        caption = item["grounding"]["caption"]

        if not regions:
            continue

        img_path = os.path.join(args.image_dir, filename)
        if not os.path.exists(img_path):
            continue

        # Official AerialVG protocol: use only the first GT box (boxes[0])
        gt_xyxy_pixel = regions[0]["bbox"]  # [x1, y1, x2, y2] in pixels
        x1, y1, x2, y2 = gt_xyxy_pixel
        gt_xyxy_norm = torch.tensor([[
            x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h
        ]], device=device, dtype=torch.float32)  # [1, 4]

        # Run model
        image = Image.open(img_path).convert("RGB")
        tensor = preprocess_image(image, args.image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor, [{"caption": caption}])

        pred_logits = outputs["pred_logits"][0].sigmoid()  # [nq, T]
        pred_boxes = outputs["pred_boxes"][0]              # [nq, 4] cxcywh normalized
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).clamp(0, 1)

        # Per-query confidence: max over text tokens (official protocol)
        confidence = pred_logits.max(dim=-1).values  # [nq]

        # Top-K by confidence
        k = min(args.top_k, confidence.shape[0])
        topk_vals, topk_idx = confidence.topk(k)
        topk_boxes = pred_boxes_xyxy[topk_idx]  # [K, 4]

        # IoU with GT (single GT box, K predictions)
        iou_matrix, _ = box_iou(topk_boxes, gt_xyxy_norm)  # [K, 1]
        ious = iou_matrix.squeeze(-1)  # [K]

        first_iou = ious[0].item()
        max_iou = ious.max().item()

        n_total += 1
        first_iou_sum += first_iou
        max_iou_sum += max_iou
        if first_iou > args.iou_thresh:
            n_top1 += 1
        if max_iou > args.iou_thresh:
            n_topk += 1

    top1_acc = n_top1 / n_total * 100 if n_total > 0 else 0.0
    topk_acc = n_topk / n_total * 100 if n_total > 0 else 0.0
    avg_first_iou = first_iou_sum / n_total if n_total > 0 else 0.0
    avg_max_iou = max_iou_sum / n_total if n_total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"AerialVG Official Top-K Accuracy Results")
    print(f"{'='*60}")
    print(f"  Checkpoint:        {args.checkpoint}")
    print(f"  Samples evaluated: {n_total}")
    print(f"  IoU threshold:     > {args.iou_thresh}")
    print(f"")
    print(f"  Top-1 Acc:         {top1_acc:.2f}%  ({n_top1}/{n_total})")
    print(f"  Top-{args.top_k} Acc:         {topk_acc:.2f}%  ({n_topk}/{n_total})")
    print(f"  Avg first IoU:     {avg_first_iou:.4f}")
    print(f"  Avg max IoU (K={args.top_k}): {avg_max_iou:.4f}")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(
        os.path.dirname(args.checkpoint), "eval_topk_results.json"
    )
    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_total": n_total,
        "n_top1": n_top1,
        "n_topk": n_topk,
        "top_1_acc": top1_acc,
        f"top_{args.top_k}_acc": topk_acc,
        "avg_first_iou": avg_first_iou,
        "avg_max_iou": avg_max_iou,
        "iou_thresh": args.iou_thresh,
        "top_k": args.top_k,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
