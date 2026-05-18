"""
Zero-shot GDINO + LoRA on AerialVG.
No relation module — LoRA only.

LoRA targets:
  - Swin-T WindowAttention : Q, K
  - Decoder self_attn      : Q, K
  - Decoder ca_text        : Q, K
  - Decoder cross_attn     : Q only (MSDeformAttn, no K)
"""

import sys, os, json, math, torch
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

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG      = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
BACKBONE    = "./checkpoints/groundingdino_swint_ogc.pth"
TRAIN_ANNO  = "/data2/huggingface/AerialVG/annotation/vg_train_odvg.jsonl"
VAL_ANNO    = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT    = "/data2/huggingface/AerialVG/images"
OUTPUT_DIR  = "./output/lora_only"
DEVICE      = "cuda:0"

EPOCHS      = 15
BATCH_SIZE  = 4
LORA_RANK   = 8
LORA_ALPHA  = 16.0
LR          = 1e-5

TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── LoRA modules ──────────────────────────────────────────────────────────────
class QKLoRAQKV(nn.Module):
    """Swin-T WindowAttention qkv: LoRA delta on Q and K slices only."""
    def __init__(self, qkv_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        d   = x.shape[-1]
        qkv[..., :d]    = qkv[..., :d]    + (x @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        qkv[..., d:2*d] = qkv[..., d:2*d] + (x @ self.lora_k_A.T @ self.lora_k_B.T) * self.scale
        return qkv


class QKLoRAMHA(nn.Module):
    """MultiheadAttention: LoRA delta on Q and K inputs."""
    def __init__(self, mha: nn.MultiheadAttention, rank: int = 8, alpha: float = 16.0):
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
    """MSDeformAttn: LoRA delta on query input only (no K)."""
    def __init__(self, deform_attn, rank: int = 8, alpha: float = 16.0):
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
    swin_count, self_count, ca_count, cross_count = 0, 0, 0, 0

    for module in model.modules():
        # Swin-T WindowAttention Q,K
        if type(module).__name__ == "WindowAttention":
            if isinstance(module.qkv, nn.Linear):
                module.qkv = QKLoRAQKV(module.qkv, rank, alpha)
                swin_count += 1

        # Decoder self_attn Q,K
        if hasattr(module, "self_attn") and isinstance(module.self_attn, nn.MultiheadAttention):
            module.self_attn = QKLoRAMHA(module.self_attn, rank, alpha)
            self_count += 1

        # Decoder ca_text Q,K
        if hasattr(module, "ca_text") and isinstance(module.ca_text, nn.MultiheadAttention):
            module.ca_text = QKLoRAMHA(module.ca_text, rank, alpha)
            ca_count += 1

        # Decoder cross_attn Q only
        if hasattr(module, "cross_attn") and type(module.cross_attn).__name__ == "MSDeformAttn":
            module.cross_attn = QLoRAMSDeformAttn(module.cross_attn, rank, alpha)
            cross_count += 1

    print(f"LoRA 적용 완료:")
    print(f"  Swin WindowAttention Q,K : {swin_count}개")
    print(f"  Decoder self_attn  Q,K   : {self_count}개")
    print(f"  Decoder ca_text    Q,K   : {ca_count}개")
    print(f"  Decoder cross_attn Q     : {cross_count}개")


# ── Dataset ───────────────────────────────────────────────────────────────────
class AerialVGDataset(Dataset):
    def __init__(self, anno_path, img_root):
        with open(anno_path) as f:
            self.metas = [json.loads(l) for l in f if l.strip()]
        self.img_root = img_root

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta     = self.metas[idx]
        img_path = os.path.join(self.img_root, meta["filename"])
        image    = Image.open(img_path).convert("RGB")
        w, h     = image.size
        image_tensor, _ = TRANSFORM(image, None)

        anchor  = meta["grounding"]["regions"][0]
        caption = meta["grounding"]["caption"]
        gt_bbox = anchor["bbox"]

        gt_norm = torch.tensor([
            gt_bbox[0]/w, gt_bbox[1]/h,
            gt_bbox[2]/w, gt_bbox[3]/h,
        ], dtype=torch.float32)

        return image_tensor, caption, gt_norm


def collate_fn(batch):
    images, captions, gt_boxes = zip(*batch)
    return list(images), list(captions), torch.stack(gt_boxes)


# ── Helpers ───────────────────────────────────────────────────────────────────
def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou_batch(boxes_a, boxes_b):
    x1 = torch.max(boxes_a[:,0].unsqueeze(1), boxes_b[:,0].unsqueeze(0))
    y1 = torch.max(boxes_a[:,1].unsqueeze(1), boxes_b[:,1].unsqueeze(0))
    x2 = torch.min(boxes_a[:,2].unsqueeze(1), boxes_b[:,2].unsqueeze(0))
    y2 = torch.min(boxes_a[:,3].unsqueeze(1), boxes_b[:,3].unsqueeze(0))
    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    area_a = ((boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1])).unsqueeze(1)
    area_b = ((boxes_b[:,2]-boxes_b[:,0])*(boxes_b[:,3]-boxes_b[:,1])).unsqueeze(0)
    return inter / (area_a + area_b - inter + 1e-6)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    for images, captions, gt_boxes in tqdm(val_loader, desc="Val", leave=False):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
            gt = gt_boxes[0].to(DEVICE)

            out    = model(img_batch, captions=captions)
            scores = out["pred_logits"].sigmoid().max(dim=-1).values[0]  # [900]
            boxes  = cxcywh_to_xyxy(out["pred_boxes"][0])                # [900,4]

            best_idx  = scores.argmax()
            pred_box  = boxes[best_idx]
            iou       = box_iou(pred_box.tolist(), gt.tolist())
            correct  += float(iou > 0.5)
            total    += 1
        except RuntimeError:
            continue

    model.train()
    return correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.chdir(Path(__file__).parent)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading: {BACKBONE}")
    model = load_model(CONFIG, BACKBONE)

    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Apply LoRA (LoRA A,B auto-unfrozen)
    apply_all_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_ds = AerialVGDataset(TRAIN_ANNO, IMG_ROOT)
    val_ds   = AerialVGDataset(VAL_ANNO,   IMG_ROOT)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, 1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0

    IOU_THRESHOLD = 0.5  # positive sample 최소 IoU

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        skipped, attempted = 0, 0
        step_loss, step_correct, step_n = 0.0, 0, 0

        for batch_idx, (images, captions, gt_boxes) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            try:
                img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
                gt = gt_boxes.to(DEVICE)  # [bs, 4] xyxy norm

                out         = model(img_batch, captions=captions)
                pred_logits = out["pred_logits"]  # [bs, 900, 256]
                pred_boxes  = out["pred_boxes"]   # [bs, 900, 4]

                bs     = pred_logits.size(0)
                scores = pred_logits.sigmoid().max(dim=-1).values  # [bs, 900]

                with torch.no_grad():
                    boxes_xyxy = cxcywh_to_xyxy(pred_boxes.detach())  # [bs, 900, 4]

                labels = []
                skip = False
                for i in range(bs):
                    attempted += 1
                    ious = box_iou_batch(boxes_xyxy[i], gt[i].unsqueeze(0)).squeeze(-1)
                    best_iou = ious.max().item()
                    if best_iou < IOU_THRESHOLD:
                        skipped += 1
                        skip = True
                        break
                    labels.append(ious.argmax())

                if skip:
                    continue

                labels = torch.stack(labels).to(DEVICE)  # [bs]
                loss = nn.CrossEntropyLoss()(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()

                total_loss  += loss.item() * bs
                correct     += (scores.argmax(1) == labels).sum().item()
                n           += bs
                step_loss   += loss.item() * bs
                step_correct += (scores.argmax(1) == labels).sum().item()
                step_n      += bs

                if (batch_idx + 1) % 100 == 0 and step_n > 0:
                    skip_rate_so_far = skipped / attempted if attempted > 0 else 0
                    print(f"  [iter {batch_idx+1:5d}] loss: {step_loss/step_n:.4f} "
                          f"acc: {step_correct/step_n:.4f} skip: {skip_rate_so_far:.1%}", flush=True)
                    step_loss, step_correct, step_n = 0.0, 0, 0

            except RuntimeError:
                continue

        scheduler.step()
        train_loss = total_loss / n if n > 0 else 0
        train_acc  = correct / n if n > 0 else 0
        skip_rate  = skipped / attempted if attempted > 0 else 0

        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1:2d} | train loss: {train_loss:.4f} acc: {train_acc:.4f} "
              f"| val acc: {val_acc:.4f} | skip: {skip_rate:.1%} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc},
                       f"{OUTPUT_DIR}/best.pth")
            print(f"  → Saved best (val acc: {val_acc:.4f})")

        torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc},
                   f"{OUTPUT_DIR}/checkpoint_ep{epoch:03d}.pth")

    print(f"\nBest val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
