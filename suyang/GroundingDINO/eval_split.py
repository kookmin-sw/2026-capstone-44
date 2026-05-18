"""
Split evaluation: relation vs no-relation samples.
Usage:
  python eval_split.py --ckpt output/lora_only/best.pth --relation_only
  python eval_split.py --ckpt output/lora_only/best.pth --no_relation
  python eval_split.py --ckpt output/lora_only/best.pth          # both splits
"""

import sys, os, json, math, argparse, torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
import groundingdino.datasets.transforms as T

# ── LoRA (same as train_lora_only.py) ─────────────────────────────────────────
class QKLoRAQKV(nn.Module):
    def __init__(self, qkv_linear, rank=8, alpha=16.0):
        super().__init__()
        self.qkv   = qkv_linear
        d          = qkv_linear.in_features
        self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d))
        self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
        for p in self.qkv.parameters():
            p.requires_grad = False

    def forward(self, x):
        qkv = self.qkv(x)
        d   = x.shape[-1]
        qkv[..., :d]    = qkv[..., :d]    + (x @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        qkv[..., d:2*d] = qkv[..., d:2*d] + (x @ self.lora_k_A.T @ self.lora_k_B.T) * self.scale
        return qkv


class QKLoRAMHA(nn.Module):
    def __init__(self, mha, rank=8, alpha=16.0):
        super().__init__()
        self.mha   = mha
        d          = mha.embed_dim
        self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d))
        self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
        for p in self.mha.parameters():
            p.requires_grad = False

    def forward(self, query, key, value, **kwargs):
        query = query + (query @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        key   = key   + (key   @ self.lora_k_A.T @ self.lora_k_B.T) * self.scale
        return self.mha(query, key, value, **kwargs)


class QLoRAMSDeformAttn(nn.Module):
    def __init__(self, deform_attn, rank=8, alpha=16.0):
        super().__init__()
        self.deform_attn = deform_attn
        d          = deform_attn.embed_dim
        self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        for p in self.deform_attn.parameters():
            p.requires_grad = False

    def forward(self, query, reference_points, value, spatial_shapes,
                level_start_index=None, key_padding_mask=None, **kwargs):
        query = query + (query @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        return self.deform_attn(query, reference_points, value, spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=key_padding_mask, **kwargs)


def apply_all_lora(model, rank=8, alpha=16.0):
    for module in model.modules():
        if type(module).__name__ == "WindowAttention":
            if isinstance(module.qkv, nn.Linear):
                module.qkv = QKLoRAQKV(module.qkv, rank, alpha)
        if hasattr(module, "self_attn") and isinstance(module.self_attn, nn.MultiheadAttention):
            module.self_attn = QKLoRAMHA(module.self_attn, rank, alpha)
        if hasattr(module, "ca_text") and isinstance(module.ca_text, nn.MultiheadAttention):
            module.ca_text = QKLoRAMHA(module.ca_text, rank, alpha)
        if hasattr(module, "cross_attn") and type(module.cross_attn).__name__ == "MSDeformAttn":
            module.cross_attn = QLoRAMSDeformAttn(module.cross_attn, rank, alpha)


# ── Config ────────────────────────────────────────────────────────────────────
CONFIG   = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
BACKBONE = "./checkpoints/groundingdino_swint_ogc.pth"
VAL_ANNO = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT = "/data2/huggingface/AerialVG/images"
DEVICE   = "cuda:0"

TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class AerialVGEvalDataset(Dataset):
    def __init__(self, anno_path, img_root, split="all"):
        """
        split: "all" | "relation" | "no_relation"
        relation  = anchor has reference objects with 'realation' key
        no_relation = anchor only, no reference objects
        """
        with open(anno_path) as f:
            all_metas = [json.loads(l) for l in f if l.strip()]

        if split == "relation":
            self.metas = [m for m in all_metas
                          if any("realation" in r for r in m["grounding"]["regions"][1:])]
        elif split == "no_relation":
            self.metas = [m for m in all_metas
                          if not any("realation" in r for r in m["grounding"]["regions"][1:])]
        else:
            self.metas = all_metas

        self.img_root = img_root
        print(f"  [{split}] {len(self.metas)} samples")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta     = self.metas[idx]
        image    = Image.open(os.path.join(self.img_root, meta["filename"])).convert("RGB")
        w, h     = image.size
        image_t, _ = TRANSFORM(image, None)
        anchor   = meta["grounding"]["regions"][0]
        gt_bbox  = anchor["bbox"]
        gt_norm  = torch.tensor([
            gt_bbox[0]/w, gt_bbox[1]/h, gt_bbox[2]/w, gt_bbox[3]/h,
        ], dtype=torch.float32)
        return image_t, meta["grounding"]["caption"], gt_norm


def collate_fn(batch):
    imgs, caps, gts = zip(*batch)
    return list(imgs), list(caps), torch.stack(gts)


# ── Helpers ───────────────────────────────────────────────────────────────────
def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, desc=""):
    model.eval()
    correct, total, errors = 0, 0, 0
    for images, captions, gt_boxes in tqdm(loader, desc=desc, leave=False):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
            gt = gt_boxes[0].to(DEVICE)

            out    = model(img_batch, captions=captions)
            scores = out["pred_logits"].sigmoid().max(dim=-1).values[0]  # [900]
            boxes  = cxcywh_to_xyxy(out["pred_boxes"][0])               # [900,4]

            best_idx = scores.argmax()
            iou      = box_iou(boxes[best_idx].tolist(), gt.tolist())
            correct += float(iou > 0.5)
            total   += 1
        except RuntimeError:
            errors += 1
            continue

    acc = correct / total if total > 0 else 0.0
    return acc, total, errors


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          type=str, required=True)
    parser.add_argument("--relation_only", action="store_true")
    parser.add_argument("--no_relation",   action="store_true")
    parser.add_argument("--device",        type=str, default=DEVICE)
    parser.add_argument("--lora_rank",     type=int, default=8)
    parser.add_argument("--lora_alpha",    type=float, default=16.0)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    device = args.device

    # Determine which splits to run
    if args.relation_only:
        splits = ["relation"]
    elif args.no_relation:
        splits = ["no_relation"]
    else:
        splits = ["all", "relation", "no_relation"]

    # Load model
    print(f"Loading backbone: {BACKBONE}")
    model = load_model(CONFIG, BACKBONE)
    model.relation_weight = 0.0
    apply_all_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
    if "val_acc" in ckpt:
        print(f"  checkpoint val_acc: {ckpt['val_acc']:.4f}  epoch: {ckpt.get('epoch', '?')}")

    model = model.to(device).eval()

    print()
    results = {}
    for split in splits:
        ds     = AerialVGEvalDataset(VAL_ANNO, IMG_ROOT, split=split)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)
        acc, total, errors = evaluate(model, loader, desc=split)
        results[split] = (acc, total, errors)

    # Report
    print("\n" + "="*55)
    print(f"{'Split':<15} {'N':>6}  {'Acc@0.5':>8}  {'Errors':>6}")
    print("-"*55)
    for split, (acc, total, errors) in results.items():
        print(f"{split:<15} {total:>6}  {acc:>8.4f}  {errors:>6}")
    print("="*55)

    if "relation" in results and "no_relation" in results:
        rel_acc   = results["relation"][0]
        norel_acc = results["no_relation"][0]
        print(f"\nRelation gap: {rel_acc - norel_acc:+.4f} "
              f"({'relation harder' if rel_acc < norel_acc else 'relation easier'})")


if __name__ == "__main__":
    main()
