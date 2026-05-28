#!/usr/bin/env python3
"""
Training script: Grounding DINO + CCM + GPQ on AerialVG.

CCM picks num_select per density class; GPQ gradually prunes the least-useful
tgt_embed slots over the first `prune_epochs` epochs (default 4 of 12),
shrinking the effective query budget from 600 → target_active (default 400).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_gpq_ccm.py \
        --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
        --pretrained_weights groundingdino/weights/groundingdino_swinb_cogcoor.pth \
        --output_dir outputs/output_GPQ_CCM \
        --num_epochs 12 --batch_size 8
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
    parser = argparse.ArgumentParser("G-DINO + CCM + GPQ Training on AerialVG")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--pretrained_weights", default=None,
                        help="Base G-DINO checkpoint to initialize from")
    parser.add_argument("--data_dir", default="/data2/huggingface/AerialVG")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
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

    # GPQ-specific
    parser.add_argument("--target_active", type=int, default=400,
                        help="Final active slot count after pruning (max query budget).")
    parser.add_argument("--prune_total", type=int, default=200,
                        help="Number of slots to prune (initial 600 → 600 - prune_total).")
    parser.add_argument("--prune_epochs", type=int, default=4,
                        help="Number of epochs over which pruning happens.")
    parser.add_argument("--prune_start_epoch", type=int, default=4,
                        help="Epoch to start pruning (Tier 2). "
                             "Score collection starts at epoch 0; actual prune deferred.")
    parser.add_argument("--score_method", type=str, default="iou",
                        choices=["confidence", "iou"],
                        help="GPQ score signal: 'iou' (Tier 2) or 'confidence' (v1).")
    parser.add_argument("--min_usage_for_prune", type=int, default=50,
                        help="Skip slots with usage_count < this when picking prune target.")
    parser.add_argument("--dynamic_query_list", type=int, nargs="+",
                        default=[200, 300, 450, 600],
                        help="CCM density→query mapping (original CCM values; "
                             "GPQ caps high-density via active_mask).")
    return parser.parse_args()


def build_optimizer(model, args):
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    bert_params    = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    other_params   = [p for n, p in model.named_parameters()
                      if "backbone" not in n and "bert" not in n and p.requires_grad]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": bert_params,     "lr": args.lr_bert},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch):
    total_steps  = args.num_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    if args.lr_drop is not None:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_drop, gamma=0.1), False
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
                    scheduler_per_iter, device, epoch, args, scaler=None,
                    global_step_start=0, prune_interval=None,
                    prune_steps_remaining=0):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.2e}"))
    header = f"Epoch [{epoch}/{args.num_epochs - 1}]"
    weight_dict = criterion.weight_dict

    raw_model = model.module if hasattr(model, "module") else model
    global_step = global_step_start

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

        global_step += 1

        # GPQ pruning (Tier 2: delayed start at args.prune_start_epoch)
        if (prune_interval is not None and prune_steps_remaining > 0
                and epoch >= args.prune_start_epoch
                and global_step % prune_interval == 0):
            raw_model.transformer.prune_one_query()
            prune_steps_remaining -= 1
            if is_main_process() and prune_steps_remaining % 20 == 0:
                num_active = raw_model.transformer.query_active_mask.sum().item()
                print(f"[GPQ] step {global_step}: active queries = {num_active}")

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()},
                             loss_total=losses.item())
        metric_logger.update(lr=optimizer.param_groups[2]["lr"])

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, global_step, prune_steps_remaining


def main():
    args = parse_args()
    init_distributed_mode(args)
    distributed = getattr(args, "distributed", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Model (CCM enabled, GPQ enabled) ──────────────────────────────────
    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = device
    model = build_model(cfg)
    model.to(device)

    # Configure CCM + GPQ on the transformer
    model.transformer.use_ccm = True
    model.transformer.dynamic_query_list = list(args.dynamic_query_list)
    model.transformer.use_gpq = True
    model.transformer.target_active_count = args.target_active

    # Tier 2 GPQ improvements
    model.transformer.gpq_score_method = args.score_method        # "iou" by default
    model.transformer.gpq_min_usage    = args.min_usage_for_prune  # 50 by default

    # Restrict GPQ pruning to slots CCM can actually use:
    # disable slots beyond max(dynamic_query_list) so they never get pruned
    # (otherwise dead slots with score=0 would be removed first, doing nothing).
    max_ccm = max(model.transformer.dynamic_query_list)
    model.transformer.query_active_mask[max_ccm:] = False

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        active_init = model.transformer.query_active_mask.sum().item()
        print(f"[CCM+GPQ] Total params: {total_params/1e6:.1f}M")
        print(f"[CCM+GPQ] dynamic_query_list = {model.transformer.dynamic_query_list}")
        print(f"[CCM+GPQ] target_active = {args.target_active} "
              f"(prune {args.prune_total} from {active_init} active slots)")
        print(f"[GPQ] dead slots [{max_ccm}..{model.transformer.num_queries}) "
              f"marked inactive — pruning operates within first {max_ccm} slots only.")
        print(f"[GPQ-Tier2] score_method={args.score_method}  "
              f"min_usage={args.min_usage_for_prune}  "
              f"prune_start_epoch={args.prune_start_epoch}")

    if args.pretrained_weights:
        model = load_pretrained(model, args.pretrained_weights)

    # ── 2. Criterion (CCM loss included) ──────────────────────────────────────
    base_weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    weight_dict = dict(base_weight_dict)
    dec_layers = getattr(cfg, "dec_layers", 6)
    for i in range(dec_layers - 1):
        for k, v in base_weight_dict.items():
            weight_dict[f"{k}_{i}"] = v
    weight_dict["loss_ccm"] = 1.0

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

    # GPQ pruning schedule
    steps_per_epoch = len(train_loader)
    total_prune_steps = args.prune_epochs * steps_per_epoch
    prune_interval = max(total_prune_steps // max(args.prune_total, 1), 1)
    prune_steps_remaining = args.prune_total
    if is_main_process():
        prune_end = args.prune_start_epoch + args.prune_epochs - 1
        print(f"[GPQ] prune {args.prune_total} slots in epochs "
              f"[{args.prune_start_epoch}..{prune_end}] ({args.prune_epochs} epoch span)")
        print(f"[GPQ] total_prune_steps={total_prune_steps}  prune_interval={prune_interval}")

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model._set_static_graph()

    # ── 5. Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(clean_state_dict(ckpt.get("model", ckpt)), strict=False)
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # Adjust GPQ remaining count from active mask
        num_active = raw_model.transformer.query_active_mask.sum().item()
        already_pruned = raw_model.transformer.num_queries - num_active
        prune_steps_remaining = max(args.prune_total - already_pruned, 0)
        if is_main_process():
            print(f"[Resume] active={num_active}, pruned={already_pruned}, remaining={prune_steps_remaining}")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    if is_main_process():
        print(f"[CCM+GPQ] Start training for {args.num_epochs} epochs")
        print(f"  Train samples: {len(train_dataset)}  |  Batch: {args.batch_size}"
              f"  |  Steps/epoch: {len(train_loader)}")

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_stats, global_step, prune_steps_remaining = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scheduler, scheduler_per_iter,
            device, epoch, args, scaler=scaler,
            global_step_start=global_step,
            prune_interval=prune_interval,
            prune_steps_remaining=prune_steps_remaining,
        )

        if not scheduler_per_iter:
            scheduler.step()

        raw_model = model.module if hasattr(model, "module") else model
        num_active = raw_model.transformer.query_active_mask.sum().item()
        checkpoint = {
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch":     epoch,
            "args":      vars(args),
            "num_active": num_active,
        }
        save_on_master(checkpoint, os.path.join(args.output_dir, "latest.pth"))
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            save_on_master(checkpoint,
                           os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pth"))

        if is_main_process():
            log_entry = {"epoch": epoch, "train": train_stats, "num_active": num_active}
            with open(os.path.join(args.output_dir, "log.json"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"Epoch {epoch:3d} | loss={train_stats.get('loss_total', float('nan')):.4f}"
                  f" | lr={train_stats.get('lr', float('nan')):.2e}"
                  f" | active={num_active}")


if __name__ == "__main__":
    main()
