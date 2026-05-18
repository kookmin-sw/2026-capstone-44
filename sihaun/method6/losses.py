from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou_pairwise


def compute_query_logits(pred_logits: torch.Tensor) -> torch.Tensor:
    if pred_logits.ndim != 3:
        raise ValueError(f"Expected pred_logits to have shape [bs, num_queries, num_tokens], got {pred_logits.shape}")
    return pred_logits.max(dim=-1).values


def pairwise_iou_cxcywh(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xyxy = box_cxcywh_to_xyxy(gt_boxes).unsqueeze(1)

    inter_x1 = torch.max(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], gt_xyxy[..., 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0.0) * (
        pred_xyxy[..., 3] - pred_xyxy[..., 1]
    ).clamp(min=0.0)
    gt_area = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp(min=0.0) * (
        gt_xyxy[..., 3] - gt_xyxy[..., 1]
    ).clamp(min=0.0)

    union = pred_area + gt_area - inter_area
    return inter_area / (union + 1e-6)


def pairwise_iou_matrix_cxcywh(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"Expected pred_boxes shape [N,4], got {tuple(pred_boxes.shape)}")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError(f"Expected target_boxes shape [M,4], got {tuple(target_boxes.shape)}")
    if target_boxes.shape[0] == 0:
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


@torch.no_grad()
def compute_topk_iou(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor):
    iou = pairwise_iou_cxcywh(pred_boxes, gt_boxes)
    first_iou = iou[:, 0]
    best_iou = iou.max(dim=1).values
    return first_iou, best_iou


@torch.no_grad()
def select_topk_boxes(outputs: dict, top_k: int):
    query_scores = compute_query_logits(outputs["pred_logits"]).sigmoid()
    k = max(1, min(top_k, query_scores.shape[1]))
    top_indices = query_scores.topk(k, dim=1).indices
    batch_indices = torch.arange(query_scores.shape[0], device=query_scores.device).unsqueeze(1)
    selected_boxes = outputs["pred_boxes"][batch_indices, top_indices]
    return selected_boxes, top_indices, query_scores


def compute_detection_losses(
    outputs: dict,
    gt_boxes: torch.Tensor,
    positive_iou: float = 0.5,
    negative_weight: float = 0.25,
    cls_loss_weight: float = 1.0,
    bbox_loss_weight: float = 5.0,
    giou_loss_weight: float = 2.0,
):
    pred_boxes = outputs["pred_boxes"]
    query_logits = compute_query_logits(outputs["pred_logits"])
    iou = pairwise_iou_cxcywh(pred_boxes, gt_boxes)

    positive_mask = iou >= positive_iou
    best_indices = iou.argmax(dim=1, keepdim=True)
    positive_mask.scatter_(1, best_indices, True)
    labels = positive_mask.float()

    bce = F.binary_cross_entropy_with_logits(query_logits, labels, reduction="none")
    zero = query_logits.sum() * 0.0

    pos_loss = bce[positive_mask].mean() if positive_mask.any() else zero
    neg_mask = ~positive_mask
    neg_loss = bce[neg_mask].mean() if neg_mask.any() else zero
    cls_loss = pos_loss + negative_weight * neg_loss

    batch_indices = torch.arange(pred_boxes.shape[0], device=pred_boxes.device)
    matched_boxes = pred_boxes[batch_indices, best_indices.squeeze(1)]
    l1_loss = F.l1_loss(matched_boxes, gt_boxes, reduction="none").sum(dim=-1).mean()

    giou = generalized_box_iou_pairwise(
        box_cxcywh_to_xyxy(matched_boxes),
        box_cxcywh_to_xyxy(gt_boxes),
    )
    giou_loss = (1.0 - giou).mean()

    total_loss = (
        cls_loss_weight * cls_loss
        + bbox_loss_weight * l1_loss
        + giou_loss_weight * giou_loss
    )

    stats = {
        "loss_total": total_loss.detach(),
        "loss_cls": cls_loss.detach(),
        "loss_bbox": l1_loss.detach(),
        "loss_giou": giou_loss.detach(),
        "mean_best_iou": iou.max(dim=1).values.mean().detach(),
        "positive_queries": labels.sum(dim=1).float().mean().detach(),
    }
    return total_loss, stats


def _resolve_selected_indices(outputs: dict, top_k: int | None = None) -> torch.Tensor:
    selected_indices = outputs.get("selected_indices")
    if selected_indices is not None:
        return selected_indices

    if "base_query_scores" in outputs:
        query_scores = outputs["base_query_scores"]
    else:
        query_scores = compute_query_logits(outputs["pred_logits"]).sigmoid()

    k = top_k if top_k is not None else min(20, query_scores.shape[1])
    k = max(1, min(k, query_scores.shape[1]))
    return query_scores.topk(k, dim=1).indices


def _unique_preserve_order(indices: List[int]) -> List[int]:
    seen = set()
    ordered = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    return ordered


def build_supervised_candidate_indices(
    outputs: dict,
    targets: List[Dict],
    top_k: int | None = None,
) -> List[torch.Tensor]:
    pred_boxes = outputs["pred_boxes"]
    selected_indices = _resolve_selected_indices(outputs, top_k=top_k)
    batch_super_indices = []

    for batch_idx, target in enumerate(targets):
        full_indices = selected_indices[batch_idx].tolist()
        image_boxes = pred_boxes[batch_idx]
        gt_box = target["gt_box"].unsqueeze(0).to(image_boxes.device)
        rescue_indices = [pairwise_iou_matrix_cxcywh(image_boxes, gt_box).squeeze(1).argmax().item()]

        aux_boxes = target.get("aux_boxes")
        if aux_boxes is not None and aux_boxes.numel() > 0:
            aux_boxes = aux_boxes.to(image_boxes.device)
            aux_iou = pairwise_iou_matrix_cxcywh(image_boxes, aux_boxes)
            rescue_indices.extend(aux_iou.argmax(dim=0).tolist())

        super_indices = _unique_preserve_order(full_indices + rescue_indices)
        batch_super_indices.append(torch.as_tensor(super_indices, dtype=torch.long, device=image_boxes.device))

    return batch_super_indices


def match_gt_and_aux_queries(
    candidate_boxes: torch.Tensor,
    gt_box: torch.Tensor,
    aux_boxes: torch.Tensor | None,
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
):
    gt_ious = pairwise_iou_matrix_cxcywh(candidate_boxes, gt_box.unsqueeze(0)).squeeze(1)
    gt_best_score, gt_best_index = gt_ious.max(dim=0)
    gt_match = int(gt_best_index.item()) if float(gt_best_score.item()) >= tau_gt else None

    aux_matches = []
    if aux_boxes is None or aux_boxes.numel() == 0:
        return gt_match, aux_matches

    aux_iou = pairwise_iou_matrix_cxcywh(candidate_boxes, aux_boxes)
    used_pred_indices = set()
    if gt_match is not None:
        used_pred_indices.add(gt_match)

    triplets = []
    for pred_idx in range(aux_iou.shape[0]):
        for aux_idx in range(aux_iou.shape[1]):
            score = float(aux_iou[pred_idx, aux_idx].item())
            if score >= tau_aux:
                triplets.append((score, pred_idx, aux_idx))

    triplets.sort(key=lambda item: item[0], reverse=True)
    used_aux_indices = set()
    for score, pred_idx, aux_idx in triplets:
        if pred_idx in used_pred_indices or aux_idx in used_aux_indices:
            continue
        used_pred_indices.add(pred_idx)
        used_aux_indices.add(aux_idx)
        aux_matches.append({
            "aux_idx": aux_idx,
            "pred_idx": pred_idx,
            "iou": score,
        })

    aux_matches.sort(key=lambda item: item["aux_idx"])
    return gt_match, aux_matches


def build_role_targets(
    outputs: dict,
    targets: List[Dict],
    top_k: int | None = None,
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
):
    pred_boxes = outputs["pred_boxes"]
    batch_super_indices = build_supervised_candidate_indices(outputs, targets, top_k=top_k)
    role_targets = []

    for batch_idx, target in enumerate(targets):
        super_indices = batch_super_indices[batch_idx]
        candidate_boxes = pred_boxes[batch_idx, super_indices]
        gt_box = target["gt_box"].to(candidate_boxes.device)
        aux_boxes = target.get("aux_boxes")
        if aux_boxes is not None:
            aux_boxes = aux_boxes.to(candidate_boxes.device)

        gt_match, aux_matches = match_gt_and_aux_queries(
            candidate_boxes,
            gt_box,
            aux_boxes,
            tau_gt=tau_gt,
            tau_aux=tau_aux,
        )

        labels = torch.full((super_indices.shape[0],), 2, dtype=torch.long, device=candidate_boxes.device)
        if gt_match is not None:
            labels[gt_match] = 0
        for match in aux_matches:
            labels[match["pred_idx"]] = 1

        role_targets.append(
            {
                "super_indices": super_indices,
                "role_labels": labels,
                "gt_match": gt_match,
                "aux_matches": aux_matches,
            }
        )

    return role_targets


def build_selected_role_targets(
    outputs: dict,
    targets: List[Dict],
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
):
    selected_boxes = outputs.get("method2_super_boxes", outputs.get("selected_boxes"))
    if selected_boxes is None:
        raise KeyError("method2 selected_boxes are required to build role targets.")
    valid_mask = outputs.get("method2_super_valid_mask")

    role_targets = []
    for batch_idx, target in enumerate(targets):
        if valid_mask is not None:
            sample_valid_mask = valid_mask[batch_idx].bool()
            candidate_boxes = selected_boxes[batch_idx][sample_valid_mask]
        else:
            sample_valid_mask = None
            candidate_boxes = selected_boxes[batch_idx]
        gt_box = target["gt_box"].to(candidate_boxes.device)
        aux_boxes = target.get("aux_boxes")
        if aux_boxes is not None:
            aux_boxes = aux_boxes.to(candidate_boxes.device)

        gt_match, aux_matches = match_gt_and_aux_queries(
            candidate_boxes,
            gt_box,
            aux_boxes,
            tau_gt=tau_gt,
            tau_aux=tau_aux,
        )

        labels = torch.full((candidate_boxes.shape[0],), 2, dtype=torch.long, device=candidate_boxes.device)
        if gt_match is not None:
            labels[gt_match] = 0
        for match in aux_matches:
            labels[match["pred_idx"]] = 1

        role_targets.append(
            {
                "role_labels": labels,
                "gt_match": gt_match,
                "aux_matches": aux_matches,
                "valid_mask": sample_valid_mask,
            }
        )
    return role_targets


def _role_accuracy_stats_from_targets(role_logits: torch.Tensor, role_targets: List[Dict]) -> Dict[str, torch.Tensor]:
    role_correct = 0.0
    role_total = 0.0
    role_gt_aux_correct = 0.0
    role_gt_aux_total = 0.0
    role_gt_correct = 0.0
    role_gt_total = 0.0
    role_aux_correct = 0.0
    role_aux_total = 0.0

    for batch_idx, target_info in enumerate(role_targets):
        labels = target_info["role_labels"]
        valid_mask = target_info.get("valid_mask")
        sample_role_logits = role_logits[batch_idx][valid_mask] if valid_mask is not None else role_logits[batch_idx]
        predictions = sample_role_logits.argmax(dim=-1)

        matches = predictions.eq(labels)
        role_correct += float(matches.sum().item())
        role_total += float(labels.numel())

        gt_aux_mask = labels.ne(2)
        if gt_aux_mask.any():
            role_gt_aux_correct += float(matches[gt_aux_mask].sum().item())
            role_gt_aux_total += float(gt_aux_mask.sum().item())

        gt_mask = labels.eq(0)
        if gt_mask.any():
            role_gt_correct += float(matches[gt_mask].sum().item())
            role_gt_total += float(gt_mask.sum().item())

        aux_mask = labels.eq(1)
        if aux_mask.any():
            role_aux_correct += float(matches[aux_mask].sum().item())
            role_aux_total += float(aux_mask.sum().item())

    role_acc = role_correct / role_total if role_total > 0 else 0.0
    role_gt_aux_acc = role_gt_aux_correct / role_gt_aux_total if role_gt_aux_total > 0 else 0.0
    role_gt_acc = role_gt_correct / role_gt_total if role_gt_total > 0 else 0.0
    role_aux_acc = role_aux_correct / role_aux_total if role_aux_total > 0 else 0.0
    return {
        "role_correct": role_logits.new_tensor(role_correct),
        "role_total": role_logits.new_tensor(role_total),
        "role_gt_aux_correct": role_logits.new_tensor(role_gt_aux_correct),
        "role_gt_aux_total": role_logits.new_tensor(role_gt_aux_total),
        "role_gt_correct": role_logits.new_tensor(role_gt_correct),
        "role_gt_total": role_logits.new_tensor(role_gt_total),
        "role_aux_correct": role_logits.new_tensor(role_aux_correct),
        "role_aux_total": role_logits.new_tensor(role_aux_total),
        "role_acc": role_logits.new_tensor(role_acc),
        "role_gt_aux_acc": role_logits.new_tensor(role_gt_aux_acc),
        "role_gt_acc": role_logits.new_tensor(role_gt_acc),
        "role_aux_acc": role_logits.new_tensor(role_aux_acc),
    }


@torch.no_grad()
def compute_role_accuracy_stats(
    outputs: dict,
    targets: List[Dict],
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
) -> Dict[str, torch.Tensor]:
    role_logits = outputs.get("method2_role_logits")
    if role_logits is None:
        raise KeyError("method2 outputs must include role logits.")
    role_targets = build_selected_role_targets(outputs, targets, tau_gt=tau_gt, tau_aux=tau_aux)
    return _role_accuracy_stats_from_targets(role_logits, role_targets)


def compute_role_and_aux_losses(
    outputs: dict,
    targets: List[Dict],
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
    role_gt_weight: float = 2.0,
    role_aux_weight: float = 2.0,
    role_none_weight: float = 0.2,
):
    role_logits = outputs.get("method2_role_logits")
    aux_attn_weights = outputs.get("method2_aux_attn_weights")
    if role_logits is None or aux_attn_weights is None:
        raise KeyError("method2 outputs must include role logits and aux attention weights.")

    role_targets = build_selected_role_targets(outputs, targets, tau_gt=tau_gt, tau_aux=tau_aux)
    zero = role_logits.sum() * 0.0

    role_loss_terms = []
    aux_loss_terms = []
    gt_positive_count = 0
    aux_positive_count = 0
    valid_aux_samples = 0
    role_class_weights = role_logits.new_tensor(
        [role_gt_weight, role_aux_weight, role_none_weight],
        dtype=role_logits.dtype,
    )

    for batch_idx, target_info in enumerate(role_targets):
        labels = target_info["role_labels"]
        valid_mask = target_info.get("valid_mask")
        sample_role_logits = role_logits[batch_idx][valid_mask] if valid_mask is not None else role_logits[batch_idx]
        role_loss_terms.append(
            F.cross_entropy(
                sample_role_logits,
                labels,
                weight=role_class_weights,
                reduction="mean",
            )
        )

        gt_match = target_info["gt_match"]
        aux_matches = target_info["aux_matches"]
        if gt_match is not None:
            gt_positive_count += 1
        aux_positive_count += len(aux_matches)

        if gt_match is None or len(aux_matches) == 0:
            continue

        sample_attn = aux_attn_weights[batch_idx]
        if valid_mask is not None:
            sample_attn = sample_attn[valid_mask][:, valid_mask]
        sample_scores = sample_attn[gt_match].clone()
        sample_scores[gt_match] = float("-inf")
        log_probs = F.log_softmax(sample_scores, dim=0)
        target_indices = torch.as_tensor(
            [match["pred_idx"] for match in aux_matches],
            dtype=torch.long,
            device=log_probs.device,
        )
        aux_loss_terms.append(-log_probs[target_indices].mean())
        valid_aux_samples += 1

    role_loss = torch.stack(role_loss_terms).mean() if role_loss_terms else zero
    aux_loss = torch.stack(aux_loss_terms).mean() if aux_loss_terms else zero
    role_acc_stats = _role_accuracy_stats_from_targets(role_logits.detach(), role_targets)

    aux_stats = {
        "loss_role": role_loss.detach(),
        "loss_aux": aux_loss.detach(),
        "gt_positive_selected": role_logits.new_tensor(float(gt_positive_count)),
        "aux_positive_selected": role_logits.new_tensor(float(aux_positive_count)),
        "valid_aux_samples": role_logits.new_tensor(float(valid_aux_samples)),
    }
    aux_stats.update(role_acc_stats)
    return role_loss, aux_loss, aux_stats


def compute_method6_losses(
    outputs: dict,
    targets: List[Dict],
    positive_iou: float = 0.5,
    negative_weight: float = 0.25,
    cls_loss_weight: float = 1.0,
    bbox_loss_weight: float = 5.0,
    giou_loss_weight: float = 2.0,
    det_loss_weight: float = 1.0,
    tau_gt: float = 0.5,
    tau_aux: float = 0.6,
    role_loss_weight: float = 0.5,
    aux_loss_weight: float = 0.0,
    role_gt_weight: float = 2.0,
    role_aux_weight: float = 2.0,
    role_none_weight: float = 0.2,
):
    gt_boxes = torch.stack([target["gt_box"] for target in targets], dim=0).to(outputs["pred_boxes"].device)
    det_loss, det_stats = compute_detection_losses(
        outputs,
        gt_boxes,
        positive_iou=positive_iou,
        negative_weight=negative_weight,
        cls_loss_weight=cls_loss_weight,
        bbox_loss_weight=bbox_loss_weight,
        giou_loss_weight=giou_loss_weight,
    )
    role_loss, aux_loss, aux_stats = compute_role_and_aux_losses(
        outputs,
        targets,
        tau_gt=tau_gt,
        tau_aux=tau_aux,
        role_gt_weight=role_gt_weight,
        role_aux_weight=role_aux_weight,
        role_none_weight=role_none_weight,
    )
    total_loss = det_loss_weight * det_loss + role_loss_weight * role_loss + aux_loss_weight * aux_loss

    stats = dict(det_stats)
    stats.update(aux_stats)
    stats["loss_det"] = det_loss.detach()
    stats["loss_total"] = total_loss.detach()
    return total_loss, stats


def compute_method2_losses(*args, **kwargs):
    return compute_method6_losses(*args, **kwargs)
