#!/usr/bin/env python3
"""
Baseline training script: Grounding DINO on AerialVG WITHOUT CCM.
Fixed 900 queries, no CCM loss — pure fine-tuning baseline.

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_base.py \
        --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
        --pretrained_weights groundingdino/models/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth \
        --output_dir outputs/train_base
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import (
    MetricLogger,
    SmoothedValue,
    collate_fn,
    init_distributed_mode,
    is_main_process,
    save_on_master,
)
from groundingdino.util.get_tokenlizer import get_tokenlizer
from groundingdino.util.utils import clean_state_dict, targets_to
from groundingdino.datasets.aerial_dataset import AerialVGDataset, build_train_transforms
from groundingdino.models.GroundingDINO.criterion import HungarianMatcher, SetCriterion


def parse_args():
    parser = argparse.ArgumentParser("Grounding DINO Baseline Training on AerialVG")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--data_dir", default="/data2/huggingface/AerialVG")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_bert", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_max_norm", type=float, default=0.1)
    parser.add_argument("--lr_drop", type=int, nargs="+", default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def build_optimizer(model, args):
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    bert_params    = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters() if "backbone" not in n and "bert" not in n and p.requires_grad]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": bert_params,     "lr": args.lr_bert},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch):
    total_steps  = args.num_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    if args.lr_drop is not None:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop, gamma=0.1), False
    base_lr = args.lr
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-7 / max(base_lr, 1e-10), 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), True


def load_pretrained(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = clean_state_dict(ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if is_main_process():
        print(f"[Pretrained] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model


def train_one_epoch(model, criterion, data_loader, optimizer, scheduler,
                    scheduler_per_iter, device, epoch, args, scaler=None):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.2e}"))
    header = f"Epoch [{epoch}/{args.num_epochs - 1}]"
    weight_dict = criterion.weight_dict

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device)
        targets = targets_to(targets, device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            losses.backward()

        if args.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler_per_iter:
            scheduler.step()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()}, loss_total=losses.item())
        metric_logger.update(lr=optimizer.param_groups[2]["lr"])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    args = parse_args()
    init_distributed_mode(args)
    distributed = getattr(args, "distributed", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Model (CCM disabled) ───────────────────────────────────────────────
    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = device
    model = build_model(cfg)
    model.transformer.use_ccm = False   # disable CCM → fixed 900 queries, no counting_output
    model.to(device)

    if args.pretrained_weights:
        model = load_pretrained(model, args.pretrained_weights)

    # ── 2. Criterion (no loss_ccm) ────────────────────────────────────────────
    base_weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    weight_dict = dict(base_weight_dict)
    dec_layers = getattr(cfg, "dec_layers", 6)
    for i in range(dec_layers - 1):
        for k, v in base_weight_dict.items():
            weight_dict[f"{k}_{i}"] = v
    # loss_ccm intentionally omitted

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=0.25, focal_gamma=2.0).to(device)

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    tokenizer  = get_tokenlizer(cfg.text_encoder_type)
    image_dir  = os.path.join(args.data_dir, "images")
    max_text_len = getattr(cfg, "max_text_len", 256)

    if is_main_process():
        print(f"Loading dataset from {args.data_dir} ...")
    raw_ds = load_dataset(args.data_dir)
    train_key = "train" if "train" in raw_ds else list(raw_ds.keys())[0]

    train_dataset = AerialVGDataset(
        hf_split=raw_ds[train_key],
        image_dir=image_dir,
        tokenizer=tokenizer,
        transforms=build_train_transforms(),
        max_text_len=max_text_len,
    )

    train_sampler = DistributedSampler(train_dataset) if distributed else RandomSampler(train_dataset)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ── 4. Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = build_optimizer(model, args)
    scheduler, scheduler_per_iter = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model._set_static_graph()

    # ── 5. Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)))
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # ── 6. Training loop ──────────────────────────────────────────────────────
    if is_main_process():
        print(f"[Baseline] Start training for {args.num_epochs} epochs (CCM disabled, fixed 900 queries)")
        print(f"  Train samples: {len(train_dataset)}  |  Batch: {args.batch_size}  |  Steps/epoch: {len(train_loader)}")

    for epoch in range(start_epoch, args.num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scheduler, scheduler_per_iter,
            device, epoch, args, scaler=scaler,
        )

        if not scheduler_per_iter:
            scheduler.step()

        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch":     epoch,
            "args":      vars(args),
        }
        save_on_master(checkpoint, os.path.join(args.output_dir, "latest.pth"))
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            save_on_master(checkpoint, os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pth"))

        if is_main_process():
            log_entry = {"epoch": epoch, "train": train_stats}
            with open(os.path.join(args.output_dir, "log.json"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"Epoch {epoch:3d} | loss={train_stats.get('loss_total', float('nan')):.4f} | lr={train_stats.get('lr', float('nan')):.2e}")


if __name__ == "__main__":
    main()
