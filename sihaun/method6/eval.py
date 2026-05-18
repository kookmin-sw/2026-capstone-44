from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from groundingdino.util.misc import collate_fn
from .aerialvg_dataset import build_aerialvg_dataset
from .losses import compute_detection_losses, compute_role_accuracy_stats, compute_topk_iou, select_topk_boxes
from .model_loader import build_method6_model, count_parameters


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Evaluate the method6 frozen-detector model on AerialVG")
    parser.add_argument("--config-file", default=None, help="Model config file. Defaults to the package-local config.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional method6 checkpoint path.")
    parser.add_argument("--strict-load", action="store_true", help="Load checkpoints with strict=True.")
    parser.add_argument("--dataset-repo", default="IPEC-COMMUNITY/AerialVG", help="Hugging Face dataset repo id.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="AerialVG split to evaluate.")
    parser.add_argument("--annotation-file", default=None, help="Optional local JSONL annotation file.")
    parser.add_argument("--image-root", default=None, help="Optional local image root.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-k boxes used for success metrics.")
    parser.add_argument("--positive-iou-threshold", type=float, default=0.5, help="Positive IoU threshold for the auxiliary eval loss.")
    parser.add_argument("--tau-gt", type=float, default=0.5, help="GT IoU threshold for method6 role matching.")
    parser.add_argument("--tau-aux", type=float, default=0.6, help="Aux IoU threshold for method6 role matching.")
    parser.add_argument(
        "--role-gate-mode",
        choices=("learned", "none", "shuffle"),
        default="learned",
        help="Role gate used for GT-Aux attention values.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on evaluated samples.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress display.")
    parser.add_argument("--disable-checkpoint", action="store_true", help="Disable activation checkpointing during evaluation.")
    parser.add_argument("--disable-transformer-ckpt", action="store_true", help="Disable transformer activation checkpointing during evaluation.")
    parser.add_argument("--log-file", default=None, help="Optional summary log file path.")
    return parser.parse_args(argv)


def move_targets_to_device(targets, device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def format_iou_threshold(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def prepare_summary_log(log_file: str | None) -> Path | None:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    print(f"Logging eval summary to {log_path}")
    return log_path


def append_summary_log(log_path: Path | None, message: str):
    if log_path is None:
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


@torch.no_grad()
def select_method6_eval_boxes(outputs: dict, top_k: int):
    return select_topk_boxes(outputs, top_k)


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    top_k: int = 5,
    positive_iou_threshold: float = 0.5,
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
    show_progress: bool = True,
):
    model.eval()
    hit_top1 = []
    hit_top5 = []
    hit_topk = []
    loss_sums = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_bbox": 0.0,
        "loss_giou": 0.0,
        "mean_best_iou": 0.0,
        "positive_queries": 0.0,
    }
    batch_count = 0
    sample_count = 0
    role_correct = 0.0
    role_total = 0.0
    role_gt_aux_correct = 0.0
    role_gt_aux_total = 0.0
    role_gt_correct = 0.0
    role_gt_total = 0.0
    role_aux_correct = 0.0
    role_aux_total = 0.0

    iterator = data_loader
    progress = None
    if show_progress:
        progress = tqdm(data_loader, total=len(data_loader), desc="Evaluating", unit="batch")
        iterator = progress

    for images, targets in iterator:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        gt_boxes = torch.stack([target["gt_box"] for target in targets], dim=0)

        outputs = model(images, targets=targets)
        _, loss_stats = compute_detection_losses(
            outputs,
            gt_boxes,
            positive_iou=positive_iou_threshold,
        )
        role_stats = compute_role_accuracy_stats(outputs, targets, tau_gt=tau_gt, tau_aux=tau_aux)
        eval_top_k = max(5, top_k)
        selected_boxes, _, _ = select_method6_eval_boxes(outputs, eval_top_k)
        first_iou, _ = compute_topk_iou(gt_boxes, selected_boxes)
        _, best_iou_top5 = compute_topk_iou(gt_boxes, selected_boxes[:, : min(5, selected_boxes.shape[1])])
        _, best_iou_topk = compute_topk_iou(gt_boxes, selected_boxes[:, : min(top_k, selected_boxes.shape[1])])

        hit_top1.append((first_iou >= positive_iou_threshold).float().cpu())
        hit_top5.append((best_iou_top5 >= positive_iou_threshold).float().cpu())
        hit_topk.append((best_iou_topk >= positive_iou_threshold).float().cpu())
        for key in loss_sums:
            loss_sums[key] += float(loss_stats[key].item())
        role_correct += float(role_stats["role_correct"].item())
        role_total += float(role_stats["role_total"].item())
        role_gt_aux_correct += float(role_stats["role_gt_aux_correct"].item())
        role_gt_aux_total += float(role_stats["role_gt_aux_total"].item())
        role_gt_correct += float(role_stats["role_gt_correct"].item())
        role_gt_total += float(role_stats["role_gt_total"].item())
        role_aux_correct += float(role_stats["role_aux_correct"].item())
        role_aux_total += float(role_stats["role_aux_total"].item())
        batch_count += 1
        sample_count += len(targets)
        if progress is not None:
            progress.set_postfix(
                samples=sample_count,
                top1_hits=int(torch.cat(hit_top1).sum().item()),
                role_acc=f"{(role_correct / role_total) if role_total > 0 else 0.0:.3f}",
                role_gt=f"{(role_gt_correct / role_gt_total) if role_gt_total > 0 else 0.0:.3f}",
                role_aux=f"{(role_aux_correct / role_aux_total) if role_aux_total > 0 else 0.0:.3f}",
            )

    if batch_count == 0:
        raise RuntimeError("No evaluation batches were produced.")

    metrics = {key: value / batch_count for key, value in loss_sums.items()}
    threshold_suffix = format_iou_threshold(positive_iou_threshold)
    top1_acc = torch.cat(hit_top1).mean().item()
    top5_acc = torch.cat(hit_top5).mean().item()
    topk_acc = torch.cat(hit_topk).mean().item()
    metrics["top1_acc"] = top1_acc
    metrics["top5_acc"] = top5_acc
    metrics[f"top{top_k}_acc"] = topk_acc
    metrics["role_correct"] = role_correct
    metrics["role_total"] = role_total
    metrics["role_gt_aux_correct"] = role_gt_aux_correct
    metrics["role_gt_aux_total"] = role_gt_aux_total
    metrics["role_gt_correct"] = role_gt_correct
    metrics["role_gt_total"] = role_gt_total
    metrics["role_aux_correct"] = role_aux_correct
    metrics["role_aux_total"] = role_aux_total
    metrics["role_acc"] = role_correct / role_total if role_total > 0 else 0.0
    metrics["role_gt_aux_acc"] = role_gt_aux_correct / role_gt_aux_total if role_gt_aux_total > 0 else 0.0
    metrics["role_gt_acc"] = role_gt_correct / role_gt_total if role_gt_total > 0 else 0.0
    metrics["role_aux_acc"] = role_aux_correct / role_aux_total if role_aux_total > 0 else 0.0
    metrics[f"top1_iou_{threshold_suffix}"] = top1_acc
    metrics[f"top5_iou_{threshold_suffix}"] = top5_acc
    metrics[f"top{top_k}_iou_{threshold_suffix}"] = topk_acc
    if positive_iou_threshold == 0.5:
        metrics["top1_iou_0.5"] = top1_acc
        metrics["top5_iou_0.5"] = top5_acc
        metrics[f"top{top_k}_iou_0.5"] = topk_acc
    metrics["num_samples"] = sample_count
    return metrics


def main(argv=None):
    args = parse_args(argv)
    summary_log_path = prepare_summary_log(args.log_file)
    dataset = build_aerialvg_dataset(
        split=args.split,
        repo_id=args.dataset_repo,
        hf_token=args.hf_token,
        revision=args.revision,
        annotation_file=args.annotation_file,
        image_root=args.image_root,
        max_samples=args.max_samples,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    model, _, load_info = build_method6_model(
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        strict=args.strict_load,
        use_checkpoint=False if args.disable_checkpoint else None,
        use_transformer_ckpt=False if args.disable_transformer_ckpt else None,
        role_gate_mode=args.role_gate_mode,
    )

    print(f"Role gate mode: {args.role_gate_mode}")
    append_summary_log(summary_log_path, f"Role gate mode: {args.role_gate_mode}")
    if load_info["checkpoint_path"] is not None:
        print(f"Loaded checkpoint: {load_info['checkpoint_path']}")
        print(f"Missing keys: {len(load_info['missing_keys'])}")
        print(f"Unexpected keys: {len(load_info['unexpected_keys'])}")
        append_summary_log(summary_log_path, f"Loaded checkpoint: {load_info['checkpoint_path']}")
    else:
        print("No checkpoint provided. Evaluating the current initialized method6 model.")
        append_summary_log(summary_log_path, "No checkpoint provided. Evaluating the current initialized method6 model.")

    if load_info["promoted_attributes"]:
        promoted = ", ".join(load_info["promoted_attributes"])
        print(f"Promoted tensor attributes to parameters: {promoted}")

    metrics = evaluate_model(
        model,
        data_loader,
        device=args.device,
        top_k=args.top_k,
        positive_iou_threshold=args.positive_iou_threshold,
        tau_gt=args.tau_gt,
        tau_aux=args.tau_aux,
        show_progress=not args.no_progress,
    )
    threshold_suffix = format_iou_threshold(args.positive_iou_threshold)
    summary_lines = [
        f"Samples evaluated: {metrics['num_samples']}",
        f"Top-1 IoU>{threshold_suffix}: {metrics['top1_acc']:.4f}",
        f"Top-5 IoU>{threshold_suffix}: {metrics['top5_acc']:.4f}",
    ]
    if args.top_k != 5:
        summary_lines.append(f"Top-{args.top_k} IoU>{threshold_suffix}: {metrics[f'top{args.top_k}_acc']:.4f}")
    summary_lines.extend(
        [
            f"Eval loss: {metrics['loss_total']:.4f}",
            f"Mean best IoU: {metrics['mean_best_iou']:.4f}",
            f"Role Acc: {metrics['role_acc']:.4f}",
            f"Role GT/Aux Acc: {metrics['role_gt_aux_acc']:.4f}",
            f"Role GT Acc: {metrics['role_gt_acc']:.4f}",
            f"Role Aux Acc: {metrics['role_aux_acc']:.4f}",
            f"Total parameters: {count_parameters(model):,}",
        ]
    )
    summary = "\n".join(summary_lines)
    print(summary)
    append_summary_log(summary_log_path, summary)


if __name__ == "__main__":
    main()
