import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from datasets import load_dataset

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from dpaa_adapter import insert_dpaa_adapters
from feature_dpaa_adapter import insert_feature_dpaa

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack(
        [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ],
        dim=-1,
    )


def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def find_image_path(local_path, filename):
    candidate_paths = [
        os.path.join(local_path, filename),
        os.path.join(local_path, "images", filename),
        os.path.join(local_path, "Images", filename),
        os.path.join(local_path, "train", filename),
        os.path.join(local_path, "validation", filename),
        os.path.join(local_path, "test", filename),
        os.path.join(local_path, "JPEGImages", filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Image file not found: {filename}\nChecked paths:\n" + "\n".join(candidate_paths)
    )


def load_groundingdino(
    config_path,
    checkpoint_path,
    device,
    use_dpaa=False,
    dpaa_mid_dim=64,
    dpaa_kernel_l=15,
    dpaa_kernel_s=3,
    dpaa_scale=1.0,
    use_feature_dpaa=False,
    feature_dpaa_mid_dim=64,
    feature_dpaa_kernel_l=15,
    feature_dpaa_kernel_s=3,
    feature_dpaa_scale=1.0,
):
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device

    model = build_model(cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)

    is_backbone_dpaa_checkpoint = any(".dpaa." in k for k in state_dict.keys())
    is_feature_dpaa_checkpoint = any("feature_dpaa" in k for k in state_dict.keys())

    if use_feature_dpaa and is_feature_dpaa_checkpoint:
        print("[Load mode] Feature-DPAA trained checkpoint")

        model = insert_feature_dpaa(
            model,
            mid_dim=feature_dpaa_mid_dim,
            kernel_l=feature_dpaa_kernel_l,
            kernel_s=feature_dpaa_kernel_s,
            scale=feature_dpaa_scale,
            verbose=False,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] missing keys: {len(missing)}")
        print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

    elif use_feature_dpaa and not is_feature_dpaa_checkpoint:
        print("[Load mode] Official checkpoint + Feature-DPAA initialized")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[Plain GroundingDINO checkpoint load]")
        print(f"[Checkpoint] missing keys: {len(missing)}")
        print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

        model = insert_feature_dpaa(
            model,
            mid_dim=feature_dpaa_mid_dim,
            kernel_l=feature_dpaa_kernel_l,
            kernel_s=feature_dpaa_kernel_s,
            scale=feature_dpaa_scale,
            verbose=False,
        )

    elif use_dpaa and is_backbone_dpaa_checkpoint:
        print("[Load mode] Backbone-DPAA trained checkpoint")

        from dpaa_adapter import insert_dpaa_adapters
        model = insert_dpaa_adapters(
            model,
            mid_dim=dpaa_mid_dim,
            kernel_l=dpaa_kernel_l,
            kernel_s=dpaa_kernel_s,
            scale=dpaa_scale,
            verbose=False,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] missing keys: {len(missing)}")
        print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

    elif use_dpaa and not is_backbone_dpaa_checkpoint:
        print("[Load mode] Official checkpoint + Backbone-DPAA initialized")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[Plain GroundingDINO checkpoint load]")
        print(f"[Checkpoint] missing keys: {len(missing)}")
        print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

        from dpaa_adapter import insert_dpaa_adapters
        model = insert_dpaa_adapters(
            model,
            mid_dim=dpaa_mid_dim,
            kernel_l=dpaa_kernel_l,
            kernel_s=dpaa_kernel_s,
            scale=dpaa_scale,
            verbose=False,
        )

    else:
        print("[Load mode] Plain GroundingDINO")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Checkpoint] missing keys: {len(missing)}")
        print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model

def preprocess_image(image, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image)


def get_topk_predictions(model, image_tensor, text, width, height, device, topk=5):
    if not text.endswith("."):
        text = text + "."

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor, captions=[text])

    pred_logits = outputs["pred_logits"].sigmoid()[0]  # [num_queries, num_tokens]
    pred_boxes = outputs["pred_boxes"][0]              # [num_queries, 4], normalized cxcywh

    scores = pred_logits.max(dim=1).values             # [num_queries]

    k = min(topk, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=k, largest=True)

    top_boxes = pred_boxes[top_indices]
    top_boxes_xyxy = box_cxcywh_to_xyxy(top_boxes)

    top_boxes_xyxy[:, 0::2] *= width
    top_boxes_xyxy[:, 1::2] *= height

    top_boxes_xyxy[:, 0::2] = top_boxes_xyxy[:, 0::2].clamp(0, width)
    top_boxes_xyxy[:, 1::2] = top_boxes_xyxy[:, 1::2].clamp(0, height)

    return top_boxes_xyxy.cpu().tolist(), top_scores.cpu().tolist()


def flatten_aerialvg(ds, use_phrase=False):
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
            bbox = region["bbox"]

            if use_phrase:
                text = phrase
            else:
                text = caption

            if not text:
                continue

            samples.append(
                {
                    "filename": filename,
                    "width": width,
                    "height": height,
                    "text": text,
                    "phrase": phrase,
                    "bbox": bbox,
                }
            )

    return samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_path", default="/data2/huggingface/AerialVG")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--use_phrase", action="store_true")
    parser.add_argument("--use_dpaa", action="store_true")
    parser.add_argument("--dpaa_mid_dim", type=int, default=64)
    parser.add_argument("--dpaa_kernel_l", type=int, default=15)
    parser.add_argument("--dpaa_kernel_s", type=int, default=3)
    parser.add_argument("--dpaa_scale", type=float, default=1.0)

    parser.add_argument("--use_feature_dpaa", action="store_true")
    parser.add_argument("--feature_dpaa_mid_dim", type=int, default=64)
    parser.add_argument("--feature_dpaa_kernel_l", type=int, default=15)
    parser.add_argument("--feature_dpaa_kernel_s", type=int, default=3)
    parser.add_argument("--feature_dpaa_scale", type=float, default=1.0)
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("Loading dataset...")
    dataset_dict = load_dataset(args.local_path)
    ds = dataset_dict[args.split]

    samples = flatten_aerialvg(ds, use_phrase=args.use_phrase)

    print(f"split: {args.split}")
    print(f"original rows: {len(ds)}")
    print(f"flattened region samples: {len(samples)}")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"evaluate samples: {len(samples)}")
    else:
        print(f"evaluate samples: all {len(samples)}")

    print("Loading model...")
    model = load_groundingdino(
    config_path=args.config,
    checkpoint_path=args.checkpoint,
    device=device,
    use_dpaa=args.use_dpaa,
    dpaa_mid_dim=args.dpaa_mid_dim,
    dpaa_kernel_l=args.dpaa_kernel_l,
    dpaa_kernel_s=args.dpaa_kernel_s,
    dpaa_scale=args.dpaa_scale,
    use_feature_dpaa=args.use_feature_dpaa,
    feature_dpaa_mid_dim=args.feature_dpaa_mid_dim,
    feature_dpaa_kernel_l=args.feature_dpaa_kernel_l,
    feature_dpaa_kernel_s=args.feature_dpaa_kernel_s,
    feature_dpaa_scale=args.feature_dpaa_scale,
)
    print("✅ 모델 로드 성공!")

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    iou_top1_sum = 0.0
    best_iou_top5_sum = 0.0

    for idx, sample in enumerate(tqdm(samples)):
        filename = sample["filename"]
        width = sample["width"]
        height = sample["height"]
        text = sample["text"]
        gt_box = sample["bbox"]

        image_path = find_image_path(args.local_path, filename)
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess_image(image, args.image_size)

        pred_boxes, scores = get_topk_predictions(
            model=model,
            image_tensor=image_tensor,
            text=text,
            width=width,
            height=height,
            device=device,
            topk=args.topk,
        )

        ious = [box_iou_xyxy(pred_box, gt_box) for pred_box in pred_boxes]

        top1_iou = ious[0]
        best_top5_iou = max(ious)

        top1_correct = top1_iou >= args.iou_thr
        top5_correct = best_top5_iou >= args.iou_thr

        correct_top1 += int(top1_correct)
        correct_top5 += int(top5_correct)
        iou_top1_sum += top1_iou
        best_iou_top5_sum += best_top5_iou
        total += 1

        if idx < 5:
            print("\n==============================")
            print(f"[{idx + 1}]")
            print("filename:", filename)
            print("text:", text)
            print("phrase:", sample["phrase"])
            print("GT box:", [round(x, 2) for x in gt_box])
            print("Top1 score:", round(scores[0], 4))
            print("Top1 pred:", [round(x, 2) for x in pred_boxes[0]])
            print("Top1 IoU:", round(top1_iou, 4))
            print("Best Top5 IoU:", round(best_top5_iou, 4))
            print("Top1 Correct:", top1_correct)
            print("Top5 Correct:", top5_correct)

    top1_acc = correct_top1 / total if total > 0 else 0.0
    top5_acc = correct_top5 / total if total > 0 else 0.0
    mean_top1_iou = iou_top1_sum / total if total > 0 else 0.0
    mean_best_top5_iou = best_iou_top5_sum / total if total > 0 else 0.0

    print("\n========== Final Result ==========")
    print(f"Setting: {'phrase' if args.use_phrase else 'caption'}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total: {total}")
    print(f"Top-1 Correct@IoU{args.iou_thr}: {correct_top1}")
    print(f"Top-1 Accuracy@{args.iou_thr}: {top1_acc * 100:.2f}%")
    print(f"Top-5 Correct@IoU{args.iou_thr}: {correct_top5}")
    print(f"Top-5 Accuracy@{args.iou_thr}: {top5_acc * 100:.2f}%")
    print(f"Mean Top-1 IoU: {mean_top1_iou:.4f}")
    print(f"Mean Best Top-5 IoU: {mean_best_top5_iou:.4f}")


if __name__ == "__main__":
    main()