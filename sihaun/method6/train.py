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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util.misc import collate_fn
from .aerialvg_dataset import build_aerialvg_dataset
from .eval import evaluate_model, move_targets_to_device
from .losses import compute_method6_losses
from .model_loader import (
    DEFAULT_TRAINABLE_PATTERNS,
    build_method6_model,
    configure_trainable_parameters,
    count_parameters,
)


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR") or os.environ.get("PY_COLORS") == "1":
        return True
    if os.environ.get("TMUX") and os.environ.get("TERM", "") != "dumb":
        return True
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def colorize(text: str, *codes: str) -> str:
    if not supports_color() or not codes:
        return text
    return "".join(codes) + text + ANSI_RESET


def print_banner(message: str, *codes: str):
    print(colorize(message, *codes), flush=True)


def format_iou_threshold(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def maybe_cuda_sync(device: str, enabled: bool, label: str | None = None):
    if not enabled:
        return
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    if label:
        print(f"[cuda-sync] {label}")


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


def stage_progress_label(stage_name: str, stage_epoch: int, stage_total_epochs: int) -> str:
    if stage_name == "single":
        return f"Single-Stage {stage_epoch}/{stage_total_epochs}"
    if stage_name == "stage1":
        return f"Stage 1 {stage_epoch}/{stage_total_epochs}"
    if stage_name == "stage2":
        return f"Stage 2 {stage_epoch}/{stage_total_epochs}"
    return f"{stage_name} {stage_epoch}/{stage_total_epochs}"


def stage_progress_desc(stage_name: str, stage_epoch: int, stage_total_epochs: int) -> str:
    label = stage_progress_label(stage_name, stage_epoch, stage_total_epochs)
    if stage_name == "single":
        return colorize(label, ANSI_BOLD, ANSI_CYAN)
    if stage_name == "stage1":
        return colorize(label, ANSI_BOLD, ANSI_BLUE)
    if stage_name == "stage2":
        return colorize(label, ANSI_BOLD, ANSI_MAGENTA)
    return colorize(label, ANSI_BOLD)


def progress_metrics(metrics: dict, schedule: dict, batch_step: int, total_batches: int) -> str:
    parts = [
        f"L={metrics.get('loss_total', 0.0):.3f}",
        f"C={metrics.get('loss_cls', 0.0):.3f}",
        f"B={metrics.get('loss_bbox', 0.0):.4f}",
        f"G={metrics.get('loss_giou', 0.0):.3f}",
        f"D={schedule['det_weight']:.2f}",
        f"R={schedule['role_weight']:.2f}",
    ]
    if "role_acc" in metrics:
        parts.append(f"RA={metrics['role_acc']:.2f}")
    if "role_gt_aux_acc" in metrics:
        parts.append(f"RGA={metrics['role_gt_aux_acc']:.2f}")
    if "role_gt_acc" in metrics:
        parts.append(f"GT={metrics['role_gt_acc']:.2f}")
    if "role_aux_acc" in metrics:
        parts.append(f"AX={metrics['role_aux_acc']:.2f}")
    if schedule["aux_weight"] > 0:
        parts.append(f"A={schedule['aux_weight']:.2f}")
    return " ".join(parts)


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


def train_mode_with_frozen_detector(model) -> None:
    model.train()
    for name, module in model.named_children():
        if name != "role_evidence_module":
            module.eval()
    model.role_evidence_module.train()


def average_running_metrics(running: dict, batch_count: int) -> dict:
    metrics = {key: value / batch_count for key, value in running.items()}
    for prefix in ("role", "role_gt_aux", "role_gt", "role_aux"):
        correct_key = f"{prefix}_correct"
        total_key = f"{prefix}_total"
        acc_key = f"{prefix}_acc"
        if running.get(total_key, 0.0) > 0:
            metrics[acc_key] = running[correct_key] / running[total_key]
            metrics[correct_key] = running[correct_key]
            metrics[total_key] = running[total_key]
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Train the method6 frozen-detector model on AerialVG")
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
    parser.add_argument("--log-file", default=None, help="Optional log file path for mirroring CLI output.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--lr-drop-epoch", type=int, default=4, help="Optional StepLR drop epoch. Disabled when set to 0.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.1, help="Gradient clipping norm.")
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5, help="Positive IoU threshold.")
    parser.add_argument("--negative-loss-weight", type=float, default=0.25, help="Relative weight for negative query BCE.")
    parser.add_argument("--det-loss-weight", type=float, default=1.0, help="Single-stage detection loss weight.")
    parser.add_argument("--cls-loss-weight", type=float, default=2.0, help="Classification loss weight.")
    parser.add_argument("--bbox-loss-weight", type=float, default=5.0, help="L1 bbox loss weight.")
    parser.add_argument("--giou-loss-weight", type=float, default=2.0, help="GIoU loss weight.")
    parser.add_argument("--tau-gt", type=float, default=0.5, help="GT IoU threshold for method6 role matching.")
    parser.add_argument("--tau-aux", type=float, default=0.6, help="Aux IoU threshold for method6 aux matching.")
    parser.add_argument("--role-loss-weight", type=float, default=0.5, help="Method6 role loss weight.")
    parser.add_argument("--aux-loss-weight", type=float, default=0.0, help="Method6 aux-selection loss weight.")
    parser.add_argument("--role-gt-weight", type=float, default=2.0, help="Role CE class weight for GT labels.")
    parser.add_argument("--role-aux-weight", type=float, default=2.0, help="Role CE class weight for Aux labels.")
    parser.add_argument("--role-none-weight", type=float, default=0.2, help="Role CE class weight for None labels.")
    parser.add_argument(
        "--role-gate-mode",
        choices=("learned", "none", "shuffle"),
        default="learned",
        help="Role gate used for GT-Aux attention values.",
    )
    parser.add_argument("--single-stage", action="store_true", help="Disable the method6 2-stage schedule and use fixed weights instead.")
    parser.add_argument("--stage1-epochs", type=int, default=3, help="Number of epochs to run in stage1.")
    parser.add_argument("--stage2-epochs", type=int, default=12, help="Number of epochs to run in stage2.")
    parser.add_argument("--stage1-role-weight", type=float, default=1.0, help="Stage1 role loss weight.")
    parser.add_argument("--stage1-aux-weight", type=float, default=0.0, help="Stage1 aux loss weight.")
    parser.add_argument("--stage1-det-weight", type=float, default=0.0, help="Stage1 detection loss weight.")
    parser.add_argument("--stage2-role-weight", type=float, default=0.1, help="Stage2 role loss weight.")
    parser.add_argument("--stage2-aux-weight", type=float, default=0.5, help="Stage2 aux loss weight target after warmup.")
    parser.add_argument("--stage2-det-weight", type=float, default=1.0, help="Stage2 detection loss weight.")
    parser.add_argument("--stage2-warmup-ratio", type=float, default=0.1, help="Warmup ratio inside stage2 for aux loss.")
    parser.add_argument(
        "--stage1-trainable-pattern",
        nargs="+",
        default=["role_text_attn", "role_context_norm", "role_mlp"],
        help="Parameter-name substrings trained during stage1.",
    )
    parser.add_argument(
        "--stage2-trainable-pattern",
        nargs="+",
        default=["role_evidence_module"],
        help="Parameter-name substrings trained during frozen-detector stage2.",
    )
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
    parser.add_argument("--log-interval", type=int, default=10, help="Print training metrics every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument(
        "--disable-checkpoint",
        action="store_true",
        help="Disable activation checkpointing in the backbone/fusion blocks for debugging or stability testing.",
    )
    parser.add_argument(
        "--disable-transformer-ckpt",
        action="store_true",
        help="Disable activation checkpointing in transformer encoder layers for debugging or stability testing.",
    )
    parser.add_argument(
        "--cuda-sync-debug",
        action="store_true",
        help="Synchronize CUDA after major training steps to surface the real failing operation earlier.",
    )
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


def prune_epoch_checkpoints(output_dir: Path, keep_recent: int = 1):
    if keep_recent < 0:
        return
    legacy_last = output_dir / "last.pt"
    if legacy_last.exists():
        legacy_last.unlink()

    epoch_checkpoints = sorted(
        output_dir.glob("epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for checkpoint in epoch_checkpoints[keep_recent:]:
        checkpoint.unlink()


def format_metrics(prefix: str, metrics: dict):
    ordered_keys = [
        "loss_total",
        "loss_det",
        "loss_cls",
        "loss_bbox",
        "loss_giou",
        "loss_role",
        "role_acc",
        "role_gt_aux_acc",
        "role_gt_acc",
        "role_aux_acc",
        "loss_aux",
        "mean_best_iou",
        "positive_queries",
        "gt_positive_selected",
        "aux_positive_selected",
        "valid_aux_samples",
    ]
    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return f"{prefix}: " + ", ".join(parts)


def resolve_stage_epoch_counts(args):
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else args.epochs
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else args.epochs
    stage1_epochs = max(0, int(stage1_epochs))
    stage2_epochs = max(0, int(stage2_epochs))
    return stage1_epochs, stage2_epochs


def resolve_stage_schedule(args, stage: str, stage_step: int = 0, stage_total_steps: int = 1):
    if args.single_stage or stage == "single":
        return {
            "stage": "single",
            "det_weight": args.det_loss_weight,
            "role_weight": args.role_loss_weight,
            "aux_weight": args.aux_loss_weight,
        }

    if stage == "stage1":
        return {
            "stage": "stage1",
            "det_weight": args.stage1_det_weight,
            "role_weight": args.stage1_role_weight,
            "aux_weight": args.stage1_aux_weight,
        }

    stage_total_steps = max(1, stage_total_steps)
    warmup_steps = max(1, int(stage_total_steps * args.stage2_warmup_ratio))
    warmup_scale = min(1.0, float(stage_step + 1) / float(warmup_steps))
    return {
        "stage": "stage2",
        "det_weight": args.stage2_det_weight,
        "role_weight": args.stage2_role_weight,
        "aux_weight": args.stage2_aux_weight * warmup_scale,
    }


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    args,
    epoch: int,
    stage_name: str,
    stage_epoch: int,
    stage_total_epochs: int,
    stage_step_start: int,
    stage_total_steps: int,
):
    train_mode_with_frozen_detector(model)
    running = defaultdict(float)
    batch_count = 0
    stage_counts = defaultdict(int)
    total_batches = len(data_loader)
    progress = tqdm(
        total=total_batches,
        desc=stage_progress_desc(stage_name, stage_epoch, stage_total_epochs),
        unit="batch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    for step, (images, targets) in enumerate(data_loader, start=1):
        stage_step = stage_step_start + step - 1
        schedule = resolve_stage_schedule(
            args,
            stage=stage_name,
            stage_step=stage_step,
            stage_total_steps=stage_total_steps,
        )
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        maybe_cuda_sync(device, args.cuda_sync_debug, f"{stage_name} step {step} after data transfer")
        outputs = model(images, targets=targets)
        maybe_cuda_sync(device, args.cuda_sync_debug, f"{stage_name} step {step} after model forward")
        loss, loss_stats = compute_method6_losses(
            outputs,
            targets,
            positive_iou=args.positive_iou_threshold,
            negative_weight=args.negative_loss_weight,
            cls_loss_weight=args.cls_loss_weight,
            bbox_loss_weight=args.bbox_loss_weight,
            giou_loss_weight=args.giou_loss_weight,
            det_loss_weight=schedule["det_weight"],
            tau_gt=args.tau_gt,
            tau_aux=args.tau_aux,
            role_loss_weight=schedule["role_weight"],
            aux_loss_weight=schedule["aux_weight"],
            role_gt_weight=args.role_gt_weight,
            role_aux_weight=args.role_aux_weight,
            role_none_weight=args.role_none_weight,
        )
        maybe_cuda_sync(device, args.cuda_sync_debug, f"{stage_name} step {step} after loss computation")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        maybe_cuda_sync(device, args.cuda_sync_debug, f"{stage_name} step {step} after backward")
        trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if args.grad_clip_norm > 0:
            clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()
        maybe_cuda_sync(device, args.cuda_sync_debug, f"{stage_name} step {step} after optimizer step")

        for key, value in loss_stats.items():
            running[key] += float(value.item())
        batch_count += 1
        stage_counts[schedule["stage"]] += 1

        averaged = average_running_metrics(running, batch_count)
        progress.update(1)
        if step == 1 or step % args.log_interval == 0 or step == total_batches:
            metrics = progress_metrics(averaged, schedule, step, total_batches)
            progress.set_postfix_str(metrics)

    if batch_count == 0:
        progress.close()
        raise RuntimeError("No training batches were produced.")
    progress.close()
    metrics = average_running_metrics(running, batch_count)
    for stage_name, count in stage_counts.items():
        metrics[f"batches_{stage_name}"] = float(count)
    return metrics, stage_step_start + batch_count


def main(argv=None):
    args = parse_args(argv)
    summary_log_path = prepare_summary_log(args.log_file)
    set_seed(args.seed)

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

    model, _, load_info = build_method6_model(
        config_file=args.config_file,
        checkpoint_path=args.init_checkpoint,
        device=args.device,
        strict=args.strict_load,
        use_checkpoint=False if args.disable_checkpoint else None,
        use_transformer_ckpt=False if args.disable_transformer_ckpt else None,
        role_gate_mode=args.role_gate_mode,
    )
    print_cuda_resolution(args.device)
    stage1_epochs, stage2_epochs = resolve_stage_epoch_counts(args)
    initial_trainable_pattern = args.trainable_pattern
    if not args.single_stage and stage1_epochs > 0:
        initial_trainable_pattern = args.stage1_trainable_pattern
    patterns, matched_names = configure_trainable_parameters(model, initial_trainable_pattern)
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
    print(f"Role gate mode: {args.role_gate_mode}")
    print("Frozen detector: yes; only matched role/evidence modules are trainable.")
    append_summary_log(summary_log_path, f"Role gate mode: {args.role_gate_mode}")
    append_summary_log(summary_log_path, "Frozen detector: yes; only matched role/evidence modules are trainable.")
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

    base_steps_per_epoch = max(1, len(train_loader))
    stage1_steps = stage1_epochs * base_steps_per_epoch
    stage2_steps = stage2_epochs * base_steps_per_epoch
    total_steps = max(1, stage1_steps + stage2_steps) if not args.single_stage else max(1, args.epochs * base_steps_per_epoch)
    if args.single_stage:
        print_banner(
            f"Training schedule: single-stage with det={args.det_loss_weight:.3f}, "
            f"role={args.role_loss_weight:.3f}, aux={args.aux_loss_weight:.3f}, "
            f"role_class_weights=(GT={args.role_gt_weight:.3f}, Aux={args.role_aux_weight:.3f}, None={args.role_none_weight:.3f})",
            ANSI_BOLD,
            ANSI_CYAN,
        )
    else:
        print_banner(
            f"Training schedule: 2-stage, stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}, "
            f"stage1_steps={stage1_steps}, stage2_steps={stage2_steps}, "
            f"stage1(det={args.stage1_det_weight:.3f}, role={args.stage1_role_weight:.3f}, aux={args.stage1_aux_weight:.3f}), "
            f"stage2(det={args.stage2_det_weight:.3f}, role={args.stage2_role_weight:.3f}, aux->{args.stage2_aux_weight:.3f}), "
            f"role_class_weights=(GT={args.role_gt_weight:.3f}, Aux={args.role_aux_weight:.3f}, None={args.role_none_weight:.3f})",
            ANSI_BOLD,
            ANSI_CYAN,
        )

    best_metric = float("-inf")
    total_epochs = args.epochs if args.single_stage else max(0, stage1_epochs + stage2_epochs)
    if total_epochs <= 0:
        raise ValueError("Total epochs resolved to 0. Provide a positive --epochs or stage epoch counts.")

    def finalize_epoch(epoch: int, train_metrics: dict):
        nonlocal best_metric
        train_summary = format_metrics(f"Epoch {epoch} train", train_metrics)
        print_banner(train_summary, ANSI_GREEN)
        append_summary_log(summary_log_path, train_summary)
        checkpoint_metrics = {f"train_{key}": value for key, value in train_metrics.items()}

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_metrics = evaluate_model(
                model,
                eval_loader,
                device=args.device,
                top_k=args.top_k,
                positive_iou_threshold=args.positive_iou_threshold,
                tau_gt=args.tau_gt,
                tau_aux=args.tau_aux,
            )
            checkpoint_metrics.update({f"eval_{key}": value for key, value in eval_metrics.items()})
            threshold_suffix = format_iou_threshold(args.positive_iou_threshold)
            eval_summary = "\n".join(
                [
                    f"Epoch {epoch} eval:",
                    f"Top-1 IoU>{threshold_suffix}: {eval_metrics['top1_acc']:.4f}",
                    f"Top-5 IoU>{threshold_suffix}: {eval_metrics['top5_acc']:.4f}",
                    f"Eval loss: {eval_metrics['loss_total']:.4f}",
                    f"Mean best IoU: {eval_metrics['mean_best_iou']:.4f}",
                    f"Role Acc: {eval_metrics['role_acc']:.4f}",
                    f"Role GT/Aux Acc: {eval_metrics['role_gt_aux_acc']:.4f}",
                    f"Role GT Acc: {eval_metrics['role_gt_acc']:.4f}",
                    f"Role Aux Acc: {eval_metrics['role_aux_acc']:.4f}",
                ]
            )
            print_banner(
                eval_summary,
                ANSI_GREEN,
            )
            append_summary_log(summary_log_path, eval_summary)

            current_metric = eval_metrics["top1_acc"]
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, checkpoint_metrics)
                print_banner(
                    f"Saved new best checkpoint with top1_acc={best_metric:.4f}",
                    ANSI_BOLD,
                    ANSI_GREEN,
                )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, checkpoint_metrics)
            prune_epoch_checkpoints(output_dir, keep_recent=0)

        if scheduler is not None:
            scheduler.step()

    if args.single_stage:
        global_step = 0
        print_banner(f"Entering single-stage training for {total_epochs} epoch(s)", ANSI_BOLD, ANSI_CYAN)
        for epoch in range(1, total_epochs + 1):
            print_banner(
                f"Starting {stage_progress_label('single', epoch, total_epochs)}",
                ANSI_BOLD,
                ANSI_CYAN,
            )
            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                args,
                epoch=epoch,
                stage_name="single",
                stage_epoch=epoch,
                stage_total_epochs=total_epochs,
                stage_step_start=global_step,
                stage_total_steps=total_steps,
            )
            finalize_epoch(epoch, train_metrics)
    else:
        stage1_global_step = 0
        print_banner(f"Entering stage1 for {stage1_epochs} epoch(s)", ANSI_BOLD, ANSI_BLUE)
        for epoch in range(1, stage1_epochs + 1):
            print_banner(
                f"Starting {stage_progress_label('stage1', epoch, stage1_epochs)}",
                ANSI_BOLD,
                ANSI_BLUE,
            )
            absolute_epoch = epoch
            train_metrics, stage1_global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                args,
                epoch=absolute_epoch,
                stage_name="stage1",
                stage_epoch=epoch,
                stage_total_epochs=stage1_epochs,
                stage_step_start=stage1_global_step,
                stage_total_steps=stage1_steps,
            )
            finalize_epoch(absolute_epoch, train_metrics)

        if stage2_epochs > 0:
            print_banner("Transitioning to stage2...", ANSI_BOLD, ANSI_YELLOW)
            patterns, matched_names = configure_trainable_parameters(model, args.stage2_trainable_pattern)
            if not matched_names:
                raise ValueError(
                    "No stage2 trainable parameters matched the requested patterns: " + ", ".join(patterns)
                )
            trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None
            if args.lr_drop_epoch > 0:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch, gamma=0.1)
            print_banner(f"Entered stage2: trainable patterns = {', '.join(patterns)}", ANSI_BOLD, ANSI_MAGENTA)

            stage2_global_step = 0
            for local_epoch in range(1, stage2_epochs + 1):
                print_banner(
                    f"Starting {stage_progress_label('stage2', local_epoch, stage2_epochs)}",
                    ANSI_BOLD,
                    ANSI_MAGENTA,
                )
                absolute_epoch = stage1_epochs + local_epoch
                train_metrics, stage2_global_step = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    args.device,
                    args,
                    epoch=absolute_epoch,
                    stage_name="stage2",
                    stage_epoch=local_epoch,
                    stage_total_epochs=stage2_epochs,
                    stage_step_start=stage2_global_step,
                    stage_total_steps=stage2_steps,
                )
                finalize_epoch(absolute_epoch, train_metrics)
if __name__ == "__main__":
    main()
