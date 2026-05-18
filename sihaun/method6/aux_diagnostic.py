from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from groundingdino.util.misc import collate_fn
from .aerialvg_dataset import build_aerialvg_dataset
from .losses import compute_query_logits, compute_role_accuracy_stats
from .model_loader import build_method6_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Diagnose how method6 final top-k predictions cover auxiliary boxes.")
    parser.add_argument("--config-file", default=None, help="Model config file. Defaults to method6/config/default.py.")
    parser.add_argument("--checkpoint-path", default=None, help="method6 checkpoint path.")
    parser.add_argument("--strict-load", action="store_true", help="Load checkpoint with strict=True.")
    parser.add_argument("--dataset-repo", default="IPEC-COMMUNITY/AerialVG", help="Hugging Face dataset repo id.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="AerialVG split to analyze.")
    parser.add_argument("--annotation-file", default=None, help="Local annotation directory or JSONL file.")
    parser.add_argument("--image-root", default=None, help="Local image root.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--revision", default=None, help="Optional dataset revision.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-k predictions to diagnose.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for GT/Aux hit.")
    parser.add_argument("--tau-gt", type=float, default=0.5, help="GT IoU threshold for method6 role matching.")
    parser.add_argument("--tau-aux", type=float, default=0.6, help="Aux IoU threshold for method6 role matching.")
    parser.add_argument(
        "--role-gate-mode",
        choices=("learned", "none", "shuffle"),
        default="learned",
        help="Role gate used for GT-Aux attention values.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    parser.add_argument("--disable-checkpoint", action="store_true", help="Disable activation checkpointing during analysis.")
    parser.add_argument(
        "--disable-transformer-ckpt",
        action="store_true",
        help="Disable transformer activation checkpointing during analysis.",
    )
    parser.add_argument("--log-file", default=None, help="Optional summary log path.")
    return parser.parse_args(argv)


def move_targets_to_device(targets, device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def pairwise_iou_cxcywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if target_boxes.numel() == 0:
        return pred_boxes.new_zeros((pred_boxes.shape[0], 0))

    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    inter_x1 = torch.max(pred_xyxy[:, None, 0], target_xyxy[None, :, 0])
    inter_y1 = torch.max(pred_xyxy[:, None, 1], target_xyxy[None, :, 1])
    inter_x2 = torch.min(pred_xyxy[:, None, 2], target_xyxy[None, :, 2])
    inter_y2 = torch.min(pred_xyxy[:, None, 3], target_xyxy[None, :, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0.0) * (
        pred_xyxy[:, 3] - pred_xyxy[:, 1]
    ).clamp(min=0.0)
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0.0) * (
        target_xyxy[:, 3] - target_xyxy[:, 1]
    ).clamp(min=0.0)

    union = pred_area[:, None] + target_area[None, :] - inter_area
    return inter_area / (union + 1e-6)


def append_summary_log(log_path: Path | None, message: str):
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


def finite_median(values: list[float]) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return float("nan")
    mid = len(finite_values) // 2
    if len(finite_values) % 2:
        return float(finite_values[mid])
    return float((finite_values[mid - 1] + finite_values[mid]) / 2)


@torch.no_grad()
def evaluate_aux_diagnostics(
    model,
    data_loader,
    device,
    top_k: int,
    iou_threshold: float,
    tau_gt: float,
    tau_aux: float,
    show_progress: bool = True,
):
    model.eval()
    total_samples = 0
    samples_with_aux = 0
    total_aux_boxes = 0

    gt_top1_hits = 0
    gt_top5_hits = 0
    gt_topk_hits = 0
    aux_top1_hits = 0
    aux_top5_hits = 0
    aux_topk_hits = 0
    samples_with_any_aux_topk = 0

    top1_gt = 0
    top1_aux = 0
    top1_none = 0
    topk_gt_count = 0
    topk_aux_count = 0
    topk_none_count = 0

    gt_first_hit_ranks = []
    aux_first_hit_ranks = []
    aux_best_iou_topk_values = []
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
        progress = tqdm(data_loader, total=len(data_loader), desc="method6 Aux diagnostic", unit="batch")
        iterator = progress

    for images, targets in iterator:
        images = images.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = model(images, targets=targets)
        role_stats = compute_role_accuracy_stats(outputs, targets, tau_gt=tau_gt, tau_aux=tau_aux)
        role_correct += float(role_stats["role_correct"].item())
        role_total += float(role_stats["role_total"].item())
        role_gt_aux_correct += float(role_stats["role_gt_aux_correct"].item())
        role_gt_aux_total += float(role_stats["role_gt_aux_total"].item())
        role_gt_correct += float(role_stats["role_gt_correct"].item())
        role_gt_total += float(role_stats["role_gt_total"].item())
        role_aux_correct += float(role_stats["role_aux_correct"].item())
        role_aux_total += float(role_stats["role_aux_total"].item())
        pred_boxes = outputs["pred_boxes"]
        query_scores = compute_query_logits(outputs["pred_logits"]).sigmoid()
        sorted_indices = query_scores.argsort(dim=1, descending=True)

        for batch_idx, target in enumerate(targets):
            total_samples += 1
            gt_box = target["gt_box"].unsqueeze(0)
            aux_boxes = target.get("aux_boxes")
            if aux_boxes is None:
                aux_boxes = pred_boxes.new_zeros((0, 4))
            num_aux = int(aux_boxes.shape[0])
            if num_aux > 0:
                samples_with_aux += 1
                total_aux_boxes += num_aux

            sample_boxes = pred_boxes[batch_idx]
            sample_sorted_indices = sorted_indices[batch_idx]
            sample_sorted_boxes = sample_boxes[sample_sorted_indices]
            sample_topk = min(top_k, sample_sorted_boxes.shape[0])
            sample_top5 = min(5, sample_sorted_boxes.shape[0])
            topk_boxes = sample_sorted_boxes[:sample_topk]

            gt_iou_sorted = pairwise_iou_cxcywh(sample_sorted_boxes, gt_box).squeeze(1)
            gt_hit_positions = (gt_iou_sorted >= iou_threshold).nonzero(as_tuple=False).flatten()
            gt_first_rank = float(gt_hit_positions[0].item() + 1) if gt_hit_positions.numel() else float("inf")
            gt_first_hit_ranks.append(gt_first_rank)
            gt_top1_hits += int(gt_first_rank <= 1)
            gt_top5_hits += int(gt_first_rank <= 5)
            gt_topk_hits += int(gt_first_rank <= sample_topk)

            aux_iou_sorted = pairwise_iou_cxcywh(sample_sorted_boxes, aux_boxes)
            if num_aux > 0:
                aux_best_iou_topk = pairwise_iou_cxcywh(topk_boxes, aux_boxes).max(dim=0).values
                aux_best_iou_topk_values.extend(aux_best_iou_topk.detach().cpu().tolist())
                aux_topk_hits += int((aux_best_iou_topk >= iou_threshold).sum().item())
                aux_top5_iou = pairwise_iou_cxcywh(sample_sorted_boxes[:sample_top5], aux_boxes).max(dim=0).values
                aux_top5_hits += int((aux_top5_iou >= iou_threshold).sum().item())
                aux_top1_iou = pairwise_iou_cxcywh(sample_sorted_boxes[:1], aux_boxes).max(dim=0).values
                aux_top1_hits += int((aux_top1_iou >= iou_threshold).sum().item())
                samples_with_any_aux_topk += int((aux_best_iou_topk >= iou_threshold).any().item())

                for aux_idx in range(num_aux):
                    aux_hit_positions = (aux_iou_sorted[:, aux_idx] >= iou_threshold).nonzero(as_tuple=False).flatten()
                    aux_first_rank = (
                        float(aux_hit_positions[0].item() + 1) if aux_hit_positions.numel() else float("inf")
                    )
                    aux_first_hit_ranks.append(aux_first_rank)

            top1_is_gt = bool(gt_iou_sorted[0] >= iou_threshold)
            top1_is_aux = bool(num_aux > 0 and aux_iou_sorted[0].max() >= iou_threshold)
            if top1_is_gt:
                top1_gt += 1
            elif top1_is_aux:
                top1_aux += 1
            else:
                top1_none += 1

            gt_iou_topk = gt_iou_sorted[:sample_topk]
            aux_iou_topk = aux_iou_sorted[:sample_topk] if num_aux > 0 else None
            for rank_idx in range(sample_topk):
                is_gt = bool(gt_iou_topk[rank_idx] >= iou_threshold)
                is_aux = bool(aux_iou_topk is not None and aux_iou_topk[rank_idx].max() >= iou_threshold)
                if is_gt:
                    topk_gt_count += 1
                elif is_aux:
                    topk_aux_count += 1
                else:
                    topk_none_count += 1

        if progress is not None:
            aux_recall = aux_topk_hits / total_aux_boxes if total_aux_boxes > 0 else 0.0
            role_gt_aux_acc = role_gt_aux_correct / role_gt_aux_total if role_gt_aux_total > 0 else 0.0
            progress.set_postfix(
                samples=total_samples,
                aux_recall_k=f"{aux_recall:.3f}",
                role_gt_aux=f"{role_gt_aux_acc:.3f}",
            )

    topk_predictions = topk_gt_count + topk_aux_count + topk_none_count
    metrics = {
        "samples": total_samples,
        "samples_with_aux": samples_with_aux,
        "aux_boxes": total_aux_boxes,
        "gt_top1": gt_top1_hits / total_samples if total_samples else 0.0,
        "gt_top5": gt_top5_hits / total_samples if total_samples else 0.0,
        "gt_topk": gt_topk_hits / total_samples if total_samples else 0.0,
        "aux_recall_top1": aux_top1_hits / total_aux_boxes if total_aux_boxes else 0.0,
        "aux_recall_top5": aux_top5_hits / total_aux_boxes if total_aux_boxes else 0.0,
        "aux_recall_topk": aux_topk_hits / total_aux_boxes if total_aux_boxes else 0.0,
        "sample_any_aux_topk": samples_with_any_aux_topk / samples_with_aux if samples_with_aux else 0.0,
        "top1_gt_rate": top1_gt / total_samples if total_samples else 0.0,
        "top1_aux_rate": top1_aux / total_samples if total_samples else 0.0,
        "top1_none_rate": top1_none / total_samples if total_samples else 0.0,
        "avg_topk_gt_count": topk_gt_count / total_samples if total_samples else 0.0,
        "avg_topk_aux_count": topk_aux_count / total_samples if total_samples else 0.0,
        "avg_topk_none_count": topk_none_count / total_samples if total_samples else 0.0,
        "topk_gt_fraction": topk_gt_count / topk_predictions if topk_predictions else 0.0,
        "topk_aux_fraction": topk_aux_count / topk_predictions if topk_predictions else 0.0,
        "topk_none_fraction": topk_none_count / topk_predictions if topk_predictions else 0.0,
        "gt_first_rank_mean": finite_mean(gt_first_hit_ranks),
        "gt_first_rank_median": finite_median(gt_first_hit_ranks),
        "aux_first_rank_mean": finite_mean(aux_first_hit_ranks),
        "aux_first_rank_median": finite_median(aux_first_hit_ranks),
        "aux_best_iou_topk_mean": finite_mean(aux_best_iou_topk_values),
        "role_acc": role_correct / role_total if role_total > 0 else 0.0,
        "role_gt_aux_acc": role_gt_aux_correct / role_gt_aux_total if role_gt_aux_total > 0 else 0.0,
        "role_gt_acc": role_gt_correct / role_gt_total if role_gt_total > 0 else 0.0,
        "role_aux_acc": role_aux_correct / role_aux_total if role_aux_total > 0 else 0.0,
    }
    return metrics


def main(argv=None):
    args = parse_args(argv)
    log_path = Path(args.log_file) if args.log_file else None

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
    append_summary_log(log_path, f"Role gate mode: {args.role_gate_mode}")
    if load_info["checkpoint_path"] is not None:
        print(f"Loaded checkpoint: {load_info['checkpoint_path']}")
        append_summary_log(log_path, f"Loaded checkpoint: {load_info['checkpoint_path']}")

    metrics = evaluate_aux_diagnostics(
        model,
        data_loader,
        device=args.device,
        top_k=args.top_k,
        iou_threshold=args.iou_threshold,
        tau_gt=args.tau_gt,
        tau_aux=args.tau_aux,
        show_progress=not args.no_progress,
    )

    lines = [
        f"Samples evaluated: {metrics['samples']}",
        f"Samples with Aux: {metrics['samples_with_aux']}",
        f"Aux boxes evaluated: {metrics['aux_boxes']}",
        f"GT Top-1 IoU>{args.iou_threshold:g}: {metrics['gt_top1']:.4f}",
        f"GT Top-5 IoU>{args.iou_threshold:g}: {metrics['gt_top5']:.4f}",
        f"GT Top-{args.top_k} IoU>{args.iou_threshold:g}: {metrics['gt_topk']:.4f}",
        f"Aux Recall@1 IoU>{args.iou_threshold:g}: {metrics['aux_recall_top1']:.4f}",
        f"Aux Recall@5 IoU>{args.iou_threshold:g}: {metrics['aux_recall_top5']:.4f}",
        f"Aux Recall@{args.top_k} IoU>{args.iou_threshold:g}: {metrics['aux_recall_topk']:.4f}",
        f"Samples with any Aux in Top-{args.top_k}: {metrics['sample_any_aux_topk']:.4f}",
        f"Top-1 role rate GT/Aux/None: {metrics['top1_gt_rate']:.4f} / {metrics['top1_aux_rate']:.4f} / {metrics['top1_none_rate']:.4f}",
        f"Average Top-{args.top_k} role counts GT/Aux/None: {metrics['avg_topk_gt_count']:.4f} / {metrics['avg_topk_aux_count']:.4f} / {metrics['avg_topk_none_count']:.4f}",
        f"Top-{args.top_k} role fraction GT/Aux/None: {metrics['topk_gt_fraction']:.4f} / {metrics['topk_aux_fraction']:.4f} / {metrics['topk_none_fraction']:.4f}",
        f"GT first-hit rank mean/median: {metrics['gt_first_rank_mean']:.2f} / {metrics['gt_first_rank_median']:.2f}",
        f"Aux first-hit rank mean/median: {metrics['aux_first_rank_mean']:.2f} / {metrics['aux_first_rank_median']:.2f}",
        f"Mean best Aux IoU in Top-{args.top_k}: {metrics['aux_best_iou_topk_mean']:.4f}",
        f"Role Acc: {metrics['role_acc']:.4f}",
        f"Role GT/Aux Acc: {metrics['role_gt_aux_acc']:.4f}",
        f"Role GT Acc: {metrics['role_gt_acc']:.4f}",
        f"Role Aux Acc: {metrics['role_aux_acc']:.4f}",
    ]
    summary = "\n".join(lines)
    print(summary)
    append_summary_log(log_path, summary)


if __name__ == "__main__":
    main()
