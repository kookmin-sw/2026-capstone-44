"""
AerialVG eval — 수정된 모델 전용
모델 구조: groundingdino.py + transformer.py (RoPE 버전)
"""
import argparse
import json
import os
import re
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


# --------------------------------------------------------------------------- #
# Strong binary spatial-relation detection
#
# AerialVG 캡션은 거의 100%가 어떤 형태든 공간 표현을 갖고 있어서 단순
# "spatial vs not" split은 무의미함 (4720 / 3 / 4723).
# 대신 두 객체 간의 *명시적 방향 관계*("X to the left of Y")가 포함된 캡션과
# 그 외(주로 "at the top right" 같은 absolute-position 묘사)를 구분.
#
# 가설: query-to-query 2D RoPE는 query 간 *상대 위치* 회전을 부여하므로,
# 두 객체의 방향 관계가 정답 단서가 되는 strong-binary 그룹에서 baseline 대비
# 더 큰 정확도 향상을 보일 것. absolute-position 그룹에서는 차이가 작을 것.
# --------------------------------------------------------------------------- #
STRONG_RELATION_PATTERNS = [
    r"\bleft of\b", r"\bright of\b", r"\bto the left of\b", r"\bto the right of\b",
    r"\babove\b", r"\bbelow\b", r"\bunder\b", r"\bunderneath\b", r"\bbeneath\b",
    r"\bin front of\b", r"\bbehind\b",
    r"\bnext to\b", r"\bbeside\b", r"\bbetween\b", r"\bnear\b",
]
_STRONG_RE = re.compile("|".join(STRONG_RELATION_PATTERNS), flags=re.IGNORECASE)


def has_spatial_relation(caption: str) -> bool:
    """True if caption contains an explicit binary spatial relation."""
    return bool(_STRONG_RE.search(caption))

from groundingdino.models import build_groundingdino
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.misc import nested_tensor_from_tensor_list
from PIL import Image as PILImage


# --------------------------------------------------------------------------- #
# make_coco_transforms('test') 재현
# 짧은 쪽 = 800, 긴 쪽 최대 1333, 종횡비 유지
# --------------------------------------------------------------------------- #
def _resize_keep_aspect(image, size=800, max_size=1333):
    w, h = image.size
    scale = size / min(w, h)
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), PILImage.BILINEAR)


def make_test_transform():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return T.Compose([
        T.Lambda(lambda img: _resize_keep_aspect(img, size=800, max_size=1333)),
        normalize,
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ODVGAerialDataset(Dataset):
    def __init__(self, image_root, ann_path, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        self.samples = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(f"{self.image_root}/{item['filename']}").convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)

        caption = item["grounding"]["caption"]
        orig_w, orig_h = item["width"], item["height"]

        region = item["grounding"]["regions"][0]
        x1, y1, x2, y2 = region["bbox"]
        cx = ((x1 + x2) / 2) / orig_w
        cy = ((y1 + y2) / 2) / orig_h
        w  = (x2 - x1) / orig_w
        h  = (y2 - y1) / orig_h

        target = {
            "anno": caption,
            "boxes": torch.tensor([[cx, cy, w, h]], dtype=torch.float32),
            "has_spatial": has_spatial_relation(caption),
        }
        return image, target


def collate_fn(batch):
    images  = nested_tensor_from_tensor_list([x[0] for x in batch])
    targets = [x[1] for x in batch]
    return images, targets


# --------------------------------------------------------------------------- #
# IoU helpers
# --------------------------------------------------------------------------- #
def convert_xyxy(box):
    new_box = torch.zeros_like(box)
    new_box[:, :, 0] = box[:, :, 0] - box[:, :, 2] / 2
    new_box[:, :, 1] = box[:, :, 1] - box[:, :, 3] / 2
    new_box[:, :, 2] = box[:, :, 0] + box[:, :, 2] / 2
    new_box[:, :, 3] = box[:, :, 1] + box[:, :, 3] / 2
    return new_box


def compute_iou(gt_box, candidate_boxes):
    gt_box = gt_box.unsqueeze(1)
    gt_box = convert_xyxy(gt_box)
    candidate_boxes = convert_xyxy(candidate_boxes)

    inter_x1 = torch.max(gt_box[:, :, 0], candidate_boxes[:, :, 0])
    inter_y1 = torch.max(gt_box[:, :, 1], candidate_boxes[:, :, 1])
    inter_x2 = torch.min(gt_box[:, :, 2], candidate_boxes[:, :, 2])
    inter_y2 = torch.min(gt_box[:, :, 3], candidate_boxes[:, :, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    gt_area   = (gt_box[:, :, 2] - gt_box[:, :, 0]) * (gt_box[:, :, 3] - gt_box[:, :, 1])
    cand_area = (candidate_boxes[:, :, 2] - candidate_boxes[:, :, 0]) * \
                (candidate_boxes[:, :, 3] - candidate_boxes[:, :, 1])
    iou = inter_area / (gt_area + cand_area - inter_area + 1e-6)

    return iou[:, 0], iou.max(dim=1).values


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
def eval_vg(model, data_loader_val, topk, device):
    """
    전체 / spatial / non-spatial 3그룹의 top-1, top-k acc@IoU>0.5 와
    샘플 단위 결과 리스트(정성 분석용)를 반환.
    """
    all_first_iou = []
    all_best_iou  = []
    all_spatial   = []        # bool per sample
    per_sample    = []        # 정성 분석용: 각 샘플의 caption, ious

    model.eval()
    model = model.to(device)

    for samples, targets in tqdm(data_loader_val, desc="Evaluating", unit="batch"):
        caption       = [t["anno"] for t in targets]
        has_spatials  = [t["has_spatial"] for t in targets]
        gt_box  = torch.stack([t["boxes"][0] for t in targets]).to(device)
        image   = samples.to(device)

        with torch.no_grad():
            outputs = model(image, captions=caption)

        logits = outputs["pred_logits"].sigmoid().max(dim=2)[0]  # (bs, nq)
        boxes  = outputs["pred_boxes"]                           # (bs, nq, 4)

        top_values, top_indices = logits.topk(topk, dim=1)
        boxes_selected = boxes[torch.arange(boxes.size(0), device=device).unsqueeze(1), top_indices]

        sorted_indices = top_values.argsort(dim=1, descending=True)
        boxes_selected_sorted = boxes_selected.gather(
            1, sorted_indices.unsqueeze(-1).expand(-1, -1, 4)
        ).to(device)

        first_iou, best_iou = compute_iou(gt_box, boxes_selected_sorted)
        all_first_iou.append(first_iou)
        all_best_iou.append(best_iou)
        all_spatial.extend(has_spatials)

        f_cpu = first_iou.detach().cpu().tolist()
        b_cpu = best_iou.detach().cpu().tolist()
        for cap, sp, fi, bi in zip(caption, has_spatials, f_cpu, b_cpu):
            per_sample.append({
                "caption": cap,
                "has_spatial": bool(sp),
                "first_iou": float(fi),
                "best_iou":  float(bi),
            })

    all_first_iou = torch.cat(all_first_iou)
    all_best_iou  = torch.cat(all_best_iou)
    spatial_mask  = torch.tensor(all_spatial, dtype=torch.bool)
    n_total       = spatial_mask.numel()
    n_spatial     = int(spatial_mask.sum().item())
    n_nonspatial  = n_total - n_spatial

    def _acc(iou, mask=None):
        if mask is not None:
            iou = iou[mask]
        if iou.numel() == 0:
            return float("nan")
        return (iou > 0.5).float().mean().item()

    results = {
        "all": {
            "n": n_total,
            "top1": _acc(all_first_iou),
            f"top{topk}": _acc(all_best_iou),
        },
        "spatial": {
            "n": n_spatial,
            "top1": _acc(all_first_iou, spatial_mask),
            f"top{topk}": _acc(all_best_iou, spatial_mask),
        },
        "non_spatial": {
            "n": n_nonspatial,
            "top1": _acc(all_first_iou, ~spatial_mask),
            f"top{topk}": _acc(all_best_iou, ~spatial_mask),
        },
    }
    return results, per_sample


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str,
                        default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--pretrain_model_path", type=str, required=True,
                        help="체크포인트 경로 (예: weights/finetuned_rope/epoch_10.pth)")
    parser.add_argument("--image_root", type=str,
                        default="/data2/huggingface/AerialVG/images")
    parser.add_argument("--ann_path", type=str,
                        default="/data2/huggingface/AerialVG/annotation/vg_test_odvg.jsonl")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_json", type=str, default=None,
                        help="결과 요약과 sample별 IoU를 JSON으로 저장 (보고서용).")
    parser.add_argument("--tag", type=str, default=None,
                        help="결과 출력/저장 시 식별용 라벨 (예: A_baseline, C_rope_v7).")
    args = parser.parse_args()

    device = torch.device(args.device)

    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_val = ODVGAerialDataset(args.image_root, args.ann_path, transforms=transform)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = str(device)

    model = build_groundingdino(cfg)
    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_res = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print(load_res)
    model.to(device)
    model.eval()

    results, per_sample = eval_vg(model, data_loader_val, topk=args.top_k, device=device)

    tag = args.tag or os.path.basename(args.pretrain_model_path)
    topk_key = f"top{args.top_k}"

    print(f"\n=== Eval result [{tag}] ===")
    print(f"ckpt: {args.pretrain_model_path}")
    print(f"{'group':<14} {'N':>6}  {'Top-1':>8}  {'Top-'+str(args.top_k):>8}")
    for group in ("all", "spatial", "non_spatial"):
        r = results[group]
        n = r["n"]
        t1 = r["top1"]
        tk = r[topk_key]
        t1_s = f"{t1*100:6.2f}%" if t1 == t1 else "   n/a"   # nan check
        tk_s = f"{tk*100:6.2f}%" if tk == tk else "   n/a"
        print(f"{group:<14} {n:>6d}  {t1_s:>8}  {tk_s:>8}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "tag":  tag,
            "ckpt": args.pretrain_model_path,
            "top_k": args.top_k,
            "summary": results,
            "per_sample": per_sample,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
