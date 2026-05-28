import os
import argparse
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
from datasets import load_dataset

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from feature_dpaa_adapter import insert_feature_dpaa


def find_image_path(local_path, filename):
    candidates = [
        os.path.join(local_path, filename),
        os.path.join(local_path, "images", filename),
        os.path.join(local_path, "Images", filename),
        os.path.join(local_path, "train", filename),
        os.path.join(local_path, "validation", filename),
        os.path.join(local_path, "test", filename),
        os.path.join(local_path, "JPEGImages", filename),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Image not found: {filename}")


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h,
    ]


def normalize_gt_xyxy_to_cxcywh(box, width, height):
    x1, y1, x2, y2 = box
    x1 /= width
    x2 /= width
    y1 /= height
    y2 /= height

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return [cx, cy, w, h]


def normalized_xyxy_to_pixel(box_xyxy, width, height):
    x1, y1, x2, y2 = box_xyxy
    return [
        int(max(0, min(width - 1, x1 * width))),
        int(max(0, min(height - 1, y1 * height))),
        int(max(0, min(width - 1, x2 * width))),
        int(max(0, min(height - 1, y2 * height))),
    ]


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def load_groundingdino(args, device):
    cfg = SLConfig.fromfile(args.config)
    cfg.device = device

    model = build_model(cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)

    has_feature_dpaa = any("feature_dpaa" in k for k in state_dict.keys())

    if args.use_feature_dpaa:
        if has_feature_dpaa:
            print("[Load] Feature-DPAA trained checkpoint")
            model = insert_feature_dpaa(
                model,
                mid_dim=args.feature_dpaa_mid_dim,
                kernel_l=args.feature_dpaa_kernel_l,
                kernel_s=args.feature_dpaa_kernel_s,
                scale=args.feature_dpaa_scale,
                verbose=False,
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
        else:
            print("[Load] Official checkpoint + initialized Feature-DPAA")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model = insert_feature_dpaa(
                model,
                mid_dim=args.feature_dpaa_mid_dim,
                kernel_l=args.feature_dpaa_kernel_l,
                kernel_s=args.feature_dpaa_kernel_s,
                scale=args.feature_dpaa_scale,
                verbose=False,
            )
    else:
        print("[Load] Plain GroundingDINO")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()
    return model


def flatten_samples(ds, use_phrase=False):
    samples = []

    for item in ds:
        filename = item["filename"]
        width = item["width"]
        height = item["height"]
        grounding = item["grounding"]

        caption = grounding.get("caption", "")
        regions = grounding.get("regions", [])

        for region in regions:
            if "bbox" not in region:
                continue

            phrase = region.get("phrase", "")
            text = phrase if use_phrase else caption

            if not text:
                continue

            samples.append({
                "filename": filename,
                "width": width,
                "height": height,
                "text": text,
                "bbox": region["bbox"],
            })

    return samples


def draw_prediction(image, gt_box_px, pred_boxes_px, scores, ious, caption, save_path):
    draw = ImageDraw.Draw(image)

    # GT: red
    draw.rectangle(gt_box_px, outline="red", width=4)
    draw.text((gt_box_px[0], max(0, gt_box_px[1] - 18)), "GT", fill="red")

    # Top-5: green
    for i, box in enumerate(pred_boxes_px):
        color = "blue" if i == 0 else "green"
        width = 4 if i == 0 else 2

        draw.rectangle(box, outline=color, width=width)
        label = f"Top-{i+1} s={scores[i]:.3f} IoU={ious[i]:.3f}"
        draw.text((box[0], max(0, box[1] - 18)), label, fill=color)

    # caption text
    text_bg_h = 60
    draw.rectangle([0, 0, image.width, text_bg_h], fill="white")
    draw.text((10, 10), caption[:180], fill="black")

    image.save(save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_path", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output_dir", default="vis_results")

    parser.add_argument("--use_phrase", action="store_true")

    parser.add_argument("--use_feature_dpaa", action="store_true")
    parser.add_argument("--feature_dpaa_mid_dim", type=int, default=64)
    parser.add_argument("--feature_dpaa_kernel_l", type=int, default=15)
    parser.add_argument("--feature_dpaa_kernel_s", type=int, default=3)
    parser.add_argument("--feature_dpaa_scale", type=float, default=2.0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("Loading dataset...")
    ds = load_dataset(args.local_path)[args.split]
    samples = flatten_samples(ds, use_phrase=args.use_phrase)

    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    print("samples:", len(samples))

    model = load_groundingdino(args, device)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    for idx, sample in enumerate(samples):
        image_path = find_image_path(args.local_path, sample["filename"])
        image = Image.open(image_path).convert("RGB")
        original = image.copy()

        width, height = image.size

        image_tensor = transform(image).unsqueeze(0).to(device)

        text = sample["text"].strip()
        if not text.endswith("."):
            text += "."

        with torch.no_grad():
            outputs = model(image_tensor, captions=[text])

        pred_logits = outputs["pred_logits"][0].sigmoid()  # [Q, T]
        pred_boxes = outputs["pred_boxes"][0]              # [Q, 4]

        scores = pred_logits.max(dim=-1).values
        k = min(args.topk, scores.shape[0])

        top_scores, top_indices = torch.topk(scores, k=k)
        top_boxes = pred_boxes[top_indices].detach().cpu()

        gt_cxcywh = normalize_gt_xyxy_to_cxcywh(sample["bbox"], width, height)
        gt_xyxy_norm = box_cxcywh_to_xyxy(gt_cxcywh)
        gt_box_px = normalized_xyxy_to_pixel(gt_xyxy_norm, width, height)

        pred_boxes_px = []
        ious = []

        for b in top_boxes:
            b = b.tolist()
            xyxy_norm = box_cxcywh_to_xyxy(b)
            box_px = normalized_xyxy_to_pixel(xyxy_norm, width, height)
            pred_boxes_px.append(box_px)
            ious.append(compute_iou(gt_xyxy_norm, xyxy_norm))

        save_name = f"{idx:04d}_top1iou_{ious[0]:.3f}_besttop5_{max(ious):.3f}.jpg"
        save_path = os.path.join(args.output_dir, save_name)

        draw_prediction(
            original,
            gt_box_px,
            pred_boxes_px,
            top_scores.detach().cpu().tolist(),
            ious,
            text,
            save_path,
        )

        print(f"[{idx+1}/{len(samples)}] saved: {save_path}")

    print("Done. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
