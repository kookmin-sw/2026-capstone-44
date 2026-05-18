from __future__ import annotations

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
