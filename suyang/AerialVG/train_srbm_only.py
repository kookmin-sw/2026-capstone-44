"""
AerialVG Training Script (HuggingFace Dataset)

Usage (single GPU):
  conda activate aerialvg_eval
  cd /home/suyang0608/suyang/AerialVG
  python train.py

Usage (multi-GPU, 4 GPUs):
  torchrun --nproc_per_node=4 train.py --output_dir ./output/train_v2
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import datasets.transforms as T
from model import build_model
from util.misc import nested_tensor_from_tensor_list
from util.slconfig import SLConfig
from util.utils import clean_state_dict
from util.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from model.AerialVG.utils import sigmoid_focal_loss


# ─── Distributed helpers ──────────────────────────────────────────────────────

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main():
    return get_rank() == 0

def setup_dist():
    if "LOCAL_RANK" not in os.environ:
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

def cleanup_dist():
    if is_dist():
        dist.destroy_process_group()


# ─── Image transforms (same as eval) ─────────────────────────────────────────

TRANSFORM = T.Compose([
    T.RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRANSFORM_VAL = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AerialVGHFDataset(Dataset):
    """
    AerialVG Dataset.  데이터를 JSONL 파일에서 직접 읽습니다.
    HuggingFace load_dataset과 동일한 데이터 구조 사용.
    Each item = one (image, caption, phrase, gt_box) triplet.
    """

    def __init__(self, jsonl_path: str, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform or TRANSFORM
        self.items = []

        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                img_path = self.image_dir / sample["filename"]
                if not img_path.exists():
                    continue
                grounding = sample.get("grounding", {})
                full_caption = grounding.get("caption", "")
                if not full_caption:
                    continue
                regions = grounding.get("regions", [])
                if not regions:
                    continue
                anchor = regions[0]
                phrase = anchor.get("phrase", "").lower().strip()
                if len(phrase) < 2:
                    continue
                # original caption
                self.items.append({
                    "img_path":  str(img_path),
                    "caption":   full_caption,
                    "phrase":    phrase,
                    "gt_box":    anchor["bbox"],
                    "img_w":     sample["width"],
                    "img_h":     sample["height"],
                })

        if is_main():
            print(f"  Dataset loaded: {len(self.items)} (image, phrase) pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["img_path"]).convert("RGB")
        image_tensor, _ = self.transform(image, None)

        x1, y1, x2, y2 = item["gt_box"]
        w, h = item["img_w"], item["img_h"]
        gt_box_norm = torch.tensor([
            (x1 + x2) / 2 / w,
            (y1 + y2) / 2 / h,
            (x2 - x1) / w,
            (y2 - y1) / h,
        ], dtype=torch.float32).clamp(0.0, 1.0)

        return {
            "image":       image_tensor,
            "caption":     item["caption"],
            "phrase":      item["phrase"],
            "gt_box_norm": gt_box_norm,
        }


def collate_fn(batch):
    return {
        "images":        [item["image"] for item in batch],
        "captions":      [item["caption"] for item in batch],
        "phrases":       [item["phrase"] for item in batch],
        "gt_boxes_norm": torch.stack([item["gt_box_norm"] for item in batch]),
    }


# ─── Token span helper ────────────────────────────────────────────────────────

def find_phrase_token_span(tokenizer, caption: str, phrase: str):
    """Find the token span of phrase within the caption tokenization.
    Caption format: 'phrase .' so phrase occupies tokens [1, len-2].
    """
    cap_ids = tokenizer(caption, add_special_tokens=True)["input_ids"]
    phr_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    n = len(phr_ids)
    for i in range(1, len(cap_ids) - n + 1):
        if cap_ids[i: i + n] == phr_ids:
            return (i, i + n)
    # fallback: span from after [CLS] to before [.] token
    return (1, max(1, len(cap_ids) - 2))


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_grounding_loss(pred_boxes, pred_logits, gt_box_norm, token_span, loss_weights):
    """
    pred_boxes:  [nq, 4]          cxcywh normalized
    pred_logits: [nq, max_text_len]
    gt_box_norm: [4]               cxcywh normalized
    token_span:  (start, end)      target phrase token range
    """
    device = pred_boxes.device
    nq = pred_boxes.shape[0]
    max_text_len = pred_logits.shape[1]

    # ── Box matching ──────────────────────────────────────────────────────────
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes).clamp(0, 1)
    gt_xyxy   = box_cxcywh_to_xyxy(gt_box_norm.unsqueeze(0))
    iou_vals  = box_iou(pred_xyxy, gt_xyxy)[0].squeeze(-1)   # [nq]
    best_q    = iou_vals.argmax()

    # ── Box regression loss (on best matched query) ───────────────────────────
    l1_loss = F.l1_loss(pred_boxes[best_q], gt_box_norm, reduction="sum")

    giou_val = generalized_box_iou(
        pred_xyxy[best_q: best_q + 1],
        gt_xyxy
    )[0, 0]
    giou_loss = 1.0 - giou_val

    box_loss = loss_weights["l1"] * l1_loss + loss_weights["giou"] * giou_loss

    # ── Classification loss ───────────────────────────────────────────────────
    # Target: 1 for tokens in span of best_q, 0 for all other (query, token) pairs
    target_cls = torch.zeros(nq, max_text_len, device=device)
    span_start, span_end = token_span
    span_end = min(span_end, max_text_len)
    if span_start < span_end:
        target_cls[best_q, span_start:span_end] = 1.0

    # Mask out padding tokens (-inf logits) to avoid NaN in focal loss
    # F.binary_cross_entropy_with_logits(-inf, 0) = max(-inf,0) - (-inf)*0 = NaN
    valid_mask = pred_logits.isfinite()          # [nq, max_text_len]
    safe_logits = pred_logits.clone()
    safe_logits[~valid_mask] = 0.0               # -inf → 0 for padding (sigmoid=0.5, ignored by mask)
    safe_target = target_cls.clone()
    safe_target[~valid_mask] = 0.0               # ignore padding targets

    cls_loss = loss_weights["cls"] * sigmoid_focal_loss(
        safe_logits, safe_target,
        alpha=0.25, gamma=2.0, num_boxes=max(nq, 1)
    )

    return box_loss + cls_loss


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data_loader, device, topk=5):
    model.eval()
    all_top1_correct = []
    all_topk_correct = []

    def convert_xyxy(box):
        new_box = torch.zeros_like(box)
        new_box[..., 0] = box[..., 0] - box[..., 2] / 2
        new_box[..., 1] = box[..., 1] - box[..., 3] / 2
        new_box[..., 2] = box[..., 0] + box[..., 2] / 2
        new_box[..., 3] = box[..., 1] + box[..., 3] / 2
        return new_box

    for batch in tqdm(data_loader, desc="Eval", leave=False, disable=not is_main()):
        images       = batch["images"]
        captions     = batch["captions"]
        gt_boxes     = batch["gt_boxes_norm"].to(device)    # [bs, 4]

        samples = nested_tensor_from_tensor_list(images).to(device)
        outputs = model(samples, captions=captions)

        logits = outputs["pred_logits"].sigmoid().max(dim=-1).values  # [bs, nq]
        boxes  = outputs["pred_boxes"]                                  # [bs, nq, 4]

        # top-k selection
        _, top_idx = logits.topk(topk, dim=1)                          # [bs, k]
        boxes_k    = boxes[torch.arange(len(images)).unsqueeze(1), top_idx]  # [bs, k, 4]

        # compute IoU
        gt_xyxy = convert_xyxy(gt_boxes)                               # [bs, 4]
        boxes_xyxy = convert_xyxy(boxes_k)                             # [bs, k, 4]

        for b in range(len(images)):
            ious = []
            for k_idx in range(topk):
                pred = boxes_xyxy[b, k_idx]
                gt   = gt_xyxy[b]
                inter_x1 = max(pred[0], gt[0])
                inter_y1 = max(pred[1], gt[1])
                inter_x2 = min(pred[2], gt[2])
                inter_y2 = min(pred[3], gt[3])
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_pred = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
                area_gt   = max(0, gt[2]   - gt[0])   * max(0, gt[3]   - gt[1])
                iou = inter / (area_pred + area_gt - inter + 1e-6)
                ious.append(iou.item())
            all_top1_correct.append(float(ious[0] > 0.5))
            all_topk_correct.append(float(max(ious) > 0.5))

    if is_dist():
        t1 = torch.tensor(all_top1_correct, device=device)
        tk = torch.tensor(all_topk_correct, device=device)
        dist.all_reduce(t1); dist.all_reduce(tk)
        n  = torch.tensor(float(len(all_top1_correct)), device=device)
        dist.all_reduce(n)
        return (t1.sum() / n).item(), (tk.sum() / n).item()

    return sum(all_top1_correct) / max(len(all_top1_correct), 1), \
           sum(all_topk_correct) / max(len(all_topk_correct), 1)


# ─── Train one epoch ──────────────────────────────────────────────────────────

def train_one_epoch(model, tokenizer, data_loader, optimizer, scaler,
                    device, epoch, loss_weights, grad_clip, args):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}", disable=not is_main())
    for batch in pbar:
        images       = batch["images"]
        captions     = batch["captions"]
        phrases      = batch["phrases"]
        gt_boxes     = batch["gt_boxes_norm"].to(device)    # [bs, 4]

        samples = nested_tensor_from_tensor_list(images).to(device)
        bs = len(images)

        with autocast(enabled=args.amp):
            outputs = model(samples, captions=captions)
            pred_logits = outputs["pred_logits"]   # [bs, nq, 256]
            pred_boxes  = outputs["pred_boxes"]    # [bs, nq, 4]

            batch_loss = torch.tensor(0.0, device=device)
            for b in range(bs):
                span = find_phrase_token_span(tokenizer, captions[b], phrases[b])
                loss_b = compute_grounding_loss(
                    pred_boxes[b], pred_logits[b],
                    gt_boxes[b], span, loss_weights
                )
                batch_loss = batch_loss + loss_b
            batch_loss = batch_loss / bs

        if not torch.isfinite(batch_loss):
            if is_main():
                print(f"  [skip] non-finite loss: {batch_loss.item()}")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

        if is_main():
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

    return total_loss / max(num_batches, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="./config/config_cfg.py")
    p.add_argument("--pretrain",    default="./checkpoints/aerialvg.pth",
                   help="Pretrained checkpoint path (strict=False load)")
    p.add_argument("--data_root",   default="/data2/huggingface/AerialVG",
                   help="AerialVG dataset root (HuggingFace format)")
    p.add_argument("--image_dir",   default="/data2/huggingface/AerialVG/images")
    p.add_argument("--output_dir",  default="./output/srbm_only")
    p.add_argument("--use_srbm",    action="store_true", default=True,
                   help="replace CrossSelfRelationTransformer with SpatialRelationBiasModule")
    p.add_argument("--srbm_layers", type=int,   default=3)
    p.add_argument("--topk",        type=int,   default=15)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_drop",     type=int,   default=4)
    p.add_argument("--weight_decay",type=float, default=1e-5)
    p.add_argument("--grad_clip",   type=float, default=0.1)
    p.add_argument("--loss_l1",     type=float, default=5.0)
    p.add_argument("--loss_giou",   type=float, default=2.0)
    p.add_argument("--loss_cls",    type=float, default=1.0)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--eval_every",  type=int,   default=1)
    p.add_argument("--topk_eval",   type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      default="",
                   help="Resume from a training checkpoint")
    return p.parse_args()


def main():
    args = get_args()
    setup_dist()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    random.seed(args.seed + get_rank())

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    # ── Build model ──────────────────────────────────────────────────────────
    cfg = SLConfig.fromfile(args.config)
    cfg_dict = cfg._cfg_dict.to_dict()
    model_args = argparse.Namespace(**cfg_dict)
    # inject SRBM-related args so build_aerialvg can use them
    model_args.use_srbm    = args.use_srbm
    model_args.srbm_layers = args.srbm_layers
    model_args.topk        = args.topk

    model = build_model(model_args).to(device)
    tokenizer = model.tokenizer

    # Load pretrained weights — relation_transformer는 제외 (새 모듈로 scratch 학습)
    if args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        state_dict = clean_state_dict(ckpt.get("model", ckpt))
        # relation_transformer 가중치 제외: backbone + BERT만 로드
        state_dict = {k: v for k, v in state_dict.items() if "relation_transformer" not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main():
            print(f"Loaded pretrain (relation_transformer excluded): missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        if is_main():
            print("No pretrained checkpoint found, training from scratch.")

    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    model_without_ddp = model.module if is_dist() else model

    # ── Freeze Detection Network (backbone + text backbone + detection module) ──
    # 공지 기준: image backbone / text backbone(BERT) / detection module 파라미터 고정
    # 학습 대상: relation_transformer 전용 (새 모듈 포함)
    for name, param in model_without_ddp.named_parameters():
        if "relation_transformer" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = [(n, p) for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    if is_main():
        print(f"Trainable params ({len(trainable)}):")
        for n, p in trainable:
            print(f"  {n}: {p.shape}")

    optimizer = torch.optim.AdamW(
        [p for _, p in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp)
    start_epoch = 0

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if is_main():
            print(f"Resumed from epoch {start_epoch}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    # HuggingFace AerialVG 데이터셋 (JSONL 파일에서 직접 읽기)
    ann_dir = os.path.join(args.data_root, "annotation")
    train_ds = AerialVGHFDataset(
        os.path.join(ann_dir, "vg_train_odvg.jsonl"),
        args.image_dir, transform=TRANSFORM
    )
    val_ds = AerialVGHFDataset(
        os.path.join(ann_dir, "vg_val_odvg.jsonl"),
        args.image_dir, transform=TRANSFORM_VAL
    )

    train_sampler = DistributedSampler(train_ds) if is_dist() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    loss_weights = {
        "l1":   args.loss_l1,
        "giou": args.loss_giou,
        "cls":  args.loss_cls,
    }

    if is_main():
        print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
        print(f"Output: {args.output_dir}")
        print(f"GPUs: {get_world_size()}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params/1e6:.1f}M")

    best_top1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        if is_dist():
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, tokenizer, train_loader, optimizer, scaler,
            device, epoch, loss_weights, args.grad_clip, args
        )
        lr_scheduler.step()

        # ── Eval ─────────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            top1, topk = evaluate(model, val_loader, device, topk=args.topk_eval)
            if is_main():
                msg = (f"Epoch {epoch:3d}  loss={train_loss:.4f}  "
                       f"Top-1={top1:.4f}  Top-{args.topk_eval}={topk:.4f}")
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

                # save best
                if top1 > best_top1:
                    best_top1 = top1
                    ckpt_path = os.path.join(args.output_dir, "best.pth")
                    torch.save({"model": model_without_ddp.state_dict(), "epoch": epoch,
                                "top1": top1, "topk": topk}, ckpt_path)
                    print(f"  => Saved best checkpoint: {ckpt_path}")

        # ── Save checkpoint each epoch ────────────────────────────────────────
        if is_main():
            torch.save({
                "model":        model_without_ddp.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler":       scaler.state_dict(),
                "epoch":        epoch,
            }, os.path.join(args.output_dir, f"checkpoint_ep{epoch:03d}.pth"))

    if is_main():
        print(f"Training done. Best Top-1: {best_top1:.4f}")
    cleanup_dist()


if __name__ == "__main__":
    main()
