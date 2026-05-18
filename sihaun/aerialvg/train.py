from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util.misc import collate_fn
from .aerialvg_dataset import build_aerialvg_dataset
from .eval import evaluate_model, move_targets_to_device
from .losses import compute_detection_losses
from .model_loader import (
    DEFAULT_TRAINABLE_PATTERNS,
    build_aerialvg_model,
    configure_trainable_parameters,
    count_parameters,
)


def progress_metrics(metrics: dict):
    return {
        "L": f"{metrics.get('loss_total', 0.0):.4f}",
        "C": f"{metrics.get('loss_cls', 0.0):.4f}",
        "B": f"{metrics.get('loss_bbox', 0.0):.4f}",
        "G": f"{metrics.get('loss_giou', 0.0):.4f}",
    }


def print_cuda_resolution(requested_device: str):
    if not torch.cuda.is_available() or not str(requested_device).startswith("cuda"):
        return

    current_idx = torch.cuda.current_device()
    current_name = torch.cuda.get_device_name(current_idx)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    message = f"Resolved CUDA device: torch cuda:{current_idx} ({current_name}) from requested {requested_device}."
    if visible_devices:
        visible_entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
        if len(visible_entries) == 1:
            message += f" logical cuda:{current_idx} maps to the only visible GPU entry '{visible_entries[0]}'."
        elif current_idx < len(visible_entries):
            message += f" logical cuda:{current_idx} maps to CUDA_VISIBLE_DEVICES[{current_idx}]='{visible_entries[current_idx]}'."
        else:
            message += f" CUDA_VISIBLE_DEVICES={visible_devices}."
    print(message)


def format_iou_threshold(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def prepare_summary_log(log_file: str | None) -> Path | None:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    print(f"Logging epoch summaries to {log_path}")
    return log_path


def append_summary_log(log_path: Path | None, message: str):
    if log_path is None:
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train the AerialVG model")
    parser.add_argument("--config-file", default=None, help="Model config file. Defaults to the package-local config.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional initialization checkpoint.")
    parser.add_argument("--strict-load", action="store_true", help="Load checkpoints with strict=True.")
    parser.add_argument("--dataset-repo", default="IPEC-COMMUNITY/AerialVG", help="Hugging Face dataset repo id.")
    parser.add_argument("--annotation-file", default=None, help="Optional local JSONL annotation file.")
    parser.add_argument("--image-root", default=None, help="Optional local image root.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--train-split", default="train", choices=("train", "val", "test"), help="Dataset split used for training.")
    parser.add_argument("--eval-split", default="val", choices=("train", "val", "test"), help="Dataset split used for evaluation.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--lr-drop-epoch", type=int, default=4, help="Optional StepLR drop epoch. Disabled when set to 0.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.1, help="Gradient clipping norm.")
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5, help="Positive IoU threshold.")
    parser.add_argument("--negative-loss-weight", type=float, default=0.25, help="Relative weight for negative query BCE.")
    parser.add_argument("--cls-loss-weight", type=float, default=2.0, help="Classification loss weight.")
    parser.add_argument("--bbox-loss-weight", type=float, default=5.0, help="L1 bbox loss weight.")
    parser.add_argument("--giou-loss-weight", type=float, default=2.0, help="GIoU loss weight.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-k used during evaluation.")
    parser.add_argument(
        "--trainable-pattern",
        nargs="+",
        default=list(DEFAULT_TRAINABLE_PATTERNS),
        help="Parameter-name substrings that remain trainable.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional limit on training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional limit on evaluation samples.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run eval every N epochs.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoints every N epochs.")
    parser.add_argument("--log-file", default=None, help="Optional summary log file path.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print training metrics every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    return parser.parse_args(argv)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def format_metrics(prefix: str, metrics: dict):
    ordered_keys = [
        "loss_total",
        "loss_cls",
        "loss_bbox",
        "loss_giou",
        "mean_best_iou",
        "positive_queries",
    ]
    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return f"{prefix}: " + ", ".join(parts)


def train_one_epoch(model, data_loader, optimizer, device, args, epoch: int):
    model.train()
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    running = defaultdict(float)
    batch_count = 0

    progress = tqdm(data_loader, total=len(data_loader), desc=f"Train Epoch {epoch}", unit="batch", dynamic_ncols=True)
    for step, (images, targets) in enumerate(progress, start=1):
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        gt_boxes = torch.stack([target["boxes"][0] for target in targets], dim=0)

        outputs = model(images, targets=targets)
        loss, loss_stats = compute_detection_losses(
            outputs,
            gt_boxes,
            positive_iou=args.positive_iou_threshold,
            negative_weight=args.negative_loss_weight,
            cls_loss_weight=args.cls_loss_weight,
            bbox_loss_weight=args.bbox_loss_weight,
            giou_loss_weight=args.giou_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()

        for key, value in loss_stats.items():
            running[key] += float(value.item())
        batch_count += 1
        averaged = {key: value / batch_count for key, value in running.items()}
        if step % args.log_interval == 0 or step == len(data_loader):
            progress.set_postfix(progress_metrics(averaged), refresh=False)

    if batch_count == 0:
        raise RuntimeError("No training batches were produced.")
    return {key: value / batch_count for key, value in running.items()}


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    summary_log_path = prepare_summary_log(args.log_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_aerialvg_dataset(
        split=args.train_split,
        repo_id=args.dataset_repo,
        hf_token=args.hf_token,
        revision=args.revision,
        annotation_file=args.annotation_file,
        image_root=args.image_root,
        max_samples=args.max_train_samples,
    )
    eval_dataset = build_aerialvg_dataset(
        split=args.eval_split,
        repo_id=args.dataset_repo,
        hf_token=args.hf_token,
        revision=args.revision,
        annotation_file=args.annotation_file,
        image_root=args.image_root,
        max_samples=args.max_eval_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model, _, load_info = build_aerialvg_model(
        config_file=args.config_file,
        checkpoint_path=args.init_checkpoint,
        device=args.device,
        strict=args.strict_load,
    )
    print_cuda_resolution(args.device)
    patterns, matched_names = configure_trainable_parameters(model, args.trainable_pattern)
    if not matched_names:
        raise ValueError(
            "No trainable parameters matched the requested patterns: " + ", ".join(patterns)
        )

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_drop_epoch > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch, gamma=0.1)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print(f"Trainable patterns: {', '.join(patterns)}")
    print(f"Matched trainable tensors: {len(matched_names)}")
    print(f"First trainable names: {', '.join(matched_names[:8])}")
    if load_info["checkpoint_path"] is not None:
        print(f"Initialized from checkpoint: {load_info['checkpoint_path']}")
        print(f"Missing keys: {len(load_info['missing_keys'])}")
        print(f"Unexpected keys: {len(load_info['unexpected_keys'])}")
    if load_info["promoted_attributes"]:
        print("Promoted tensor attributes: " + ", ".join(load_info["promoted_attributes"]))

    best_metric = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, args.device, args, epoch=epoch)
        train_summary = format_metrics(f"Epoch {epoch} train", train_metrics)
        print(train_summary)
        append_summary_log(summary_log_path, train_summary)

        checkpoint_metrics = {f"train_{key}": value for key, value in train_metrics.items()}

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_metrics = evaluate_model(
                model,
                eval_loader,
                device=args.device,
                top_k=args.top_k,
                positive_iou_threshold=args.positive_iou_threshold,
            )
            checkpoint_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            threshold_suffix = format_iou_threshold(args.positive_iou_threshold)
            eval_summary = (
                f"Epoch {epoch} eval: top1_acc@{threshold_suffix}={eval_metrics['top1_acc']:.4f}, "
                f"top5_acc@{threshold_suffix}={eval_metrics['top5_acc']:.4f}"
            )
            if args.top_k != 5:
                eval_summary += f", top{args.top_k}_acc@{threshold_suffix}={eval_metrics[f'top{args.top_k}_acc']:.4f}"
            eval_summary += f", loss_total={eval_metrics['loss_total']:.4f}"
            print(eval_summary)
            append_summary_log(summary_log_path, eval_summary)

            current_metric = eval_metrics["top1_acc"]
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, checkpoint_metrics)
                print(f"Saved new best checkpoint with top1_acc@{threshold_suffix}={best_metric:.4f}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, args, checkpoint_metrics)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, args, checkpoint_metrics)

        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
