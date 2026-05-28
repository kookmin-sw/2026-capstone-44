import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import box_convert, box_iou

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import clean_state_dict

from aerial_dataset import AerialVGJsonlDataset, collate_fn


def load_groundingdino(config_path, checkpoint_path, device):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess_caption(caption):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


@torch.no_grad()
def get_topk_candidates(gdino, image, caption, topk, device):
    caption = preprocess_caption(caption)
    outputs = gdino(image[None].to(device), captions=[caption])
    pred_logits = outputs["pred_logits"].sigmoid()[0]
    pred_boxes = outputs["pred_boxes"][0]
    base_scores = pred_logits.max(dim=1)[0]
    k = min(topk, base_scores.shape[0])
    topk_scores, topk_idx = torch.topk(base_scores, k=k, dim=0)
    topk_logits = pred_logits[topk_idx]
    topk_boxes = pred_boxes[topk_idx]
    return topk_logits.cpu(), topk_boxes.cpu(), topk_scores.cpu()


def assign_positive_candidate(topk_boxes, gt_box):
    pred_xyxy = box_convert(topk_boxes, "cxcywh", "xyxy")
    gt_xyxy = box_convert(gt_box.unsqueeze(0), "cxcywh", "xyxy")
    ious = box_iou(pred_xyxy, gt_xyxy).squeeze(1)
    positive_idx = torch.argmax(ious)
    return positive_idx, ious.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    gdino = load_groundingdino(args.config, args.checkpoint, args.device)
    dataset = AerialVGJsonlDataset(args.json_path, args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    cached_data = []

    for batch in tqdm(loader, desc="extract"):
        for image, caption, gt_box, filename in zip(
            batch["images"],
            batch["captions"],
            batch["gt_boxes"],
            batch["filenames"],
        ):
            gt_box = gt_box.cpu()
            image_path = os.path.join(args.image_root, filename)

            topk_logits, topk_boxes, topk_scores = get_topk_candidates(
                gdino, image, caption, args.topk, args.device,
            )

            positive_idx, max_iou = assign_positive_candidate(topk_boxes, gt_box)

            cached_data.append({
                "topk_logits":  topk_logits.half(),
                "topk_boxes":   topk_boxes.float(),
                "topk_scores":  topk_scores.float(),
                "gt_box":       gt_box.float(),
                "positive_idx": positive_idx.long(),
                "max_iou":      max_iou.float(),
                "caption":      caption,
                "image_path":   image_path,
            })

    torch.save(cached_data, args.save_path)
    print(f"Saved cached features to {args.save_path}")


if __name__ == "__main__":
    main()
