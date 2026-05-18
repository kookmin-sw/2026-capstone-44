"""
Train SRBM (Stage 2) on AerialVG.
- Stage 1 (frozen): GDINO + LoRA, loaded from LORA_CKPT
- Stage 2 (trained): SpatialRelationBiasModule (relation_v3.py)
- Loss: BCE with IoU as soft targets (IoU > 0.5 → positive)
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
from groundingdino.models.GroundingDINO.relation_v3 import SpatialRelationBiasModule

# ── LoRA (same as train_lora_only.py) ────────────────────────────────────────
import copy
from groundingdino.util.inference import load_model

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
CONFIG      = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
BACKBONE    = "./checkpoints/groundingdino_swint_ogc.pth"
LORA_CKPT   = "./output/lora_only/best.pth"
TRAIN_ANNO  = "/data2/huggingface/AerialVG/annotation/vg_train_odvg.jsonl"
VAL_ANNO    = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT    = "/data2/huggingface/AerialVG/images"
OUTPUT_DIR  = "/data2/2026_capstone/suyang0608/srbm_on_fullft"
DEVICE      = "cuda:0"

EPOCHS        = 15
BATCH_SIZE    = 4
LORA_RANK     = 8
LORA_ALPHA    = 16.0
LR            = 1e-4
TOPK          = 15
IOU_THRESHOLD = 0.1
SRBM_LAYERS   = 3
RESUME_CKPT   = None

TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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

def box_iou_batch(boxes_a, boxes_b):
    x1 = torch.max(boxes_a[:,0].unsqueeze(1), boxes_b[:,0].unsqueeze(0))
    y1 = torch.max(boxes_a[:,1].unsqueeze(1), boxes_b[:,1].unsqueeze(0))
    x2 = torch.min(boxes_a[:,2].unsqueeze(1), boxes_b[:,2].unsqueeze(0))
    y2 = torch.min(boxes_a[:,3].unsqueeze(1), boxes_b[:,3].unsqueeze(0))
    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    aa = ((boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1])).unsqueeze(1)
    ab = ((boxes_b[:,2]-boxes_b[:,0])*(boxes_b[:,3]-boxes_b[:,1])).unsqueeze(0)
    return inter / (aa + ab - inter + 1e-6)


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    srbm = model.relation_transformer
    correct, total = 0, 0

    for images, captions, gt_boxes in tqdm(val_loader, desc="Val", leave=False):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
            gt = gt_boxes[0].to(DEVICE)

            model.relation_weight = 0.0
            out = model(img_batch, captions=captions)

            cache    = model._cache
            final_hs       = cache["final_hs"].detach()
            final_pred_boxes = cache["final_pred_boxes"].detach()
            final_pred_logits = cache["final_pred_logits"].detach()
            text_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in cache["text_dict"].items()}

            scores   = final_pred_logits.sigmoid().max(dim=-1).values  # [1, 900]
            topk_idx = scores.topk(TOPK, dim=1).indices                # [1, K]

            topk_features = final_hs.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, final_hs.size(-1))
            )
            topk_coords = final_pred_boxes.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
            )

            srbm_logits = srbm(topk_features, topk_coords, text_dict)  # [1, K, max_text_len]
            srbm_scores = srbm_logits.max(dim=-1).values                 # [1, K]

            best_k   = srbm_scores[0].argmax().item()
            pred_box = cxcywh_to_xyxy(topk_coords[0])[best_k]
            iou      = box_iou(pred_box.tolist(), gt.tolist())
            correct += float(iou > 0.5)
            total   += 1
        except RuntimeError:
            continue

    model.train()
    return correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.chdir(Path(__file__).parent)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load base model
    print(f"Loading backbone: {BACKBONE}")
    model = load_model(CONFIG, BACKBONE)
    model.relation_weight = 0.0  # backbone-only forward during training

    # 2. Load Full FT checkpoint (NO LoRA — backbone is fully fine-tuned)
    FULL_FT_CKPT = "/home/suyang0608/suyang/checkpoint_baseline/checkpoint_baseline.pth"
    print(f"Loading Full FT checkpoint: {FULL_FT_CKPT}")
    ckpt = torch.load(FULL_FT_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")

    # 4. Replace relation_transformer with SRBM
    model.relation_transformer = SpatialRelationBiasModule(
        d_model=256, num_heads=8, num_layers=SRBM_LAYERS,
        topk=TOPK, max_text_len=256,
    )
    srbm_params = sum(p.numel() for p in model.relation_transformer.parameters())
    print(f"SRBM params: {srbm_params:,}  (layers={SRBM_LAYERS})")

    # 5. Freeze everything except SRBM
    for p in model.parameters():
        p.requires_grad = False
    for p in model.relation_transformer.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = model.to(DEVICE).train()

    # Datasets
    train_ds = AerialVGDataset(TRAIN_ANNO, IMG_ROOT)
    val_ds   = AerialVGDataset(VAL_ANNO,   IMG_ROOT)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds, 1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    srbm = model.relation_transformer
    best_acc  = 0.0
    start_epoch = 0

    # Resume from SRBM checkpoint if available
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        print(f"Resuming from: {RESUME_CKPT}")
        srbm_ckpt = torch.load(RESUME_CKPT, map_location="cpu")
        missing, unexpected = model.load_state_dict(srbm_ckpt["model"], strict=False)
        print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
        start_epoch = srbm_ckpt.get("epoch", 0) + 1
        best_acc    = srbm_ckpt.get("val_acc", 0.0)
        # fast-forward scheduler to match resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  resumed epoch={start_epoch}, best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        skipped, attempted = 0, 0
        step_loss, step_correct, step_n = 0.0, 0, 0

        for batch_idx, (images, captions, gt_boxes) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (cont)")
        ):
            try:
                img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
                gt = gt_boxes.to(DEVICE)  # [bs, 4] xyxy norm
                bs = gt.size(0)

                # Frozen backbone forward (no SRBM)
                model.relation_weight = 0.0
                with torch.no_grad():
                    model(img_batch, captions=captions)

                cache = model._cache
                final_hs          = cache["final_hs"].detach()
                final_pred_boxes  = cache["final_pred_boxes"].detach()
                final_pred_logits = cache["final_pred_logits"].detach()
                text_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v
                             for k, v in cache["text_dict"].items()}

                # Top-K selection
                scores   = final_pred_logits.sigmoid().max(dim=-1).values  # [bs, 900]
                topk_idx = scores.topk(TOPK, dim=1).indices                # [bs, K]

                topk_features = final_hs.gather(
                    1, topk_idx.unsqueeze(-1).expand(-1, -1, final_hs.size(-1))
                )
                topk_coords = final_pred_boxes.gather(
                    1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
                )
                topk_boxes_xyxy = cxcywh_to_xyxy(topk_coords)

                # GT IoU targets for BCE
                iou_targets = []
                skip = False
                for i in range(bs):
                    attempted += 1
                    ious = box_iou_batch(topk_boxes_xyxy[i], gt[i].unsqueeze(0)).squeeze(-1)  # [K]
                    if ious.max().item() < IOU_THRESHOLD:
                        skipped += 1
                        skip = True
                        break
                    iou_targets.append(ious)

                if skip:
                    continue

                iou_targets = torch.stack(iou_targets).to(DEVICE)  # [bs, K]
                bce_labels  = (iou_targets > 0.5).float()          # [bs, K]

                # SRBM forward (only SRBM params get gradients)
                srbm_logits = srbm(topk_features, topk_coords, text_dict)  # [bs, K, max_text_len]
                srbm_scores = srbm_logits.max(dim=-1).values                 # [bs, K]

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    srbm_scores, bce_labels
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()

                # accuracy: argmax of scores vs argmax of IoU (for monitoring)
                labels       = iou_targets.argmax(dim=1)           # [bs]
                total_loss   += loss.item() * bs
                correct      += (srbm_scores.argmax(1) == labels).sum().item()
                n            += bs
                step_loss    += loss.item() * bs
                step_correct += (srbm_scores.argmax(1) == labels).sum().item()
                step_n       += bs

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
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_acc": val_acc,
                "srbm_layers": SRBM_LAYERS,
            }, f"{OUTPUT_DIR}/best.pth")
            print(f"  → Saved best (val acc: {val_acc:.4f})")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "val_acc": val_acc,
        }, f"{OUTPUT_DIR}/checkpoint_ep{epoch:03d}.pth")

    print(f"\nBest val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
