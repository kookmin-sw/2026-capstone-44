from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss


def _fix_degenerate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Ensure x2 >= x1 and y2 >= y1 for xyxy boxes."""
    return torch.stack([
        torch.min(boxes[:, 0], boxes[:, 2]),
        torch.min(boxes[:, 1], boxes[:, 3]),
        torch.max(boxes[:, 0], boxes[:, 2]),
        torch.max(boxes[:, 1], boxes[:, 3]),
    ], dim=-1)


class HungarianMatcher(nn.Module):
    """Optimal bipartite matching between predictions and ground-truth.

    Cost matrix combines:
      - Classification cost: negative dot product between query sigmoid
        probabilities and GT positive_map (phrase token distribution).
      - L1 box cost: pairwise L1 distance in cxcywh normalized space.
      - GIoU cost: negative generalized IoU.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: dict with
                "pred_logits": [bs, num_queries, max_text_len]
                "pred_boxes":  [bs, num_queries, 4]  cxcywh normalized
            targets: list of dicts, each with
                "boxes":        [N, 4]  cxcywh normalized
                "positive_map": [N, max_text_len]  row-normalized

        Returns:
            List of (query_indices, gt_indices) tensors, one per image.
        """
        bs, nq = outputs["pred_logits"].shape[:2]
        indices = []

        for i in range(bs):
            pred_logits_i = outputs["pred_logits"][i]  # [nq, T]
            pred_boxes_i = outputs["pred_boxes"][i]     # [nq, 4]
            tgt_boxes_i = targets[i]["boxes"]           # [N, 4]
            tgt_pm_i = targets[i]["positive_map"]       # [N, T]

            N = len(tgt_boxes_i)
            if N == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # Move GT to same device as predictions
            tgt_boxes_i = tgt_boxes_i.to(pred_boxes_i.device)
            tgt_pm_i = tgt_pm_i.to(pred_logits_i.device)

            # Classification cost: -sigmoid(logits) @ positive_map.T  [nq, N]
            prob = pred_logits_i.sigmoid()
            cost_class = -(prob @ tgt_pm_i.T)

            # L1 box cost  [nq, N]
            cost_bbox = torch.cdist(pred_boxes_i, tgt_boxes_i, p=1)

            # GIoU cost  [nq, N]
            pred_xyxy = _fix_degenerate_boxes(
                box_cxcywh_to_xyxy(pred_boxes_i).clamp(0, 1)
            )
            tgt_xyxy = _fix_degenerate_boxes(
                box_cxcywh_to_xyxy(tgt_boxes_i).clamp(0, 1)
            )
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )  # [nq, N]

            row_idx, col_idx = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_idx, dtype=torch.long),
                torch.as_tensor(col_idx, dtype=torch.long),
            ))

        return indices


class SetCriterion(nn.Module):
    """Loss for Grounding DINO training.

    Combines:
      - Focal classification loss on all queries (matched positives + unmatched negatives).
      - L1 + GIoU bounding box loss on matched pairs only.
    Applied to final decoder layer output and all auxiliary intermediate outputs.
    """

    def __init__(
        self,
        matcher: HungarianMatcher,
        weight_dict: dict,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ccm_loss_fn = nn.CrossEntropyLoss()

    def _loss_labels(
        self,
        outputs: dict,
        targets: List[dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> dict:
        pred_logits = outputs["pred_logits"]  # [bs, nq, T]
        bs, nq, T = pred_logits.shape
        device = pred_logits.device

        # Build full target tensor: matched positions get positive_map values,
        # unmatched queries remain 0 (treated as all-negative by focal loss).
        tgt_full = torch.zeros(bs, nq, T, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            tgt_full[i][src_idx] = targets[i]["positive_map"][tgt_idx].to(device)

        loss_ce = sigmoid_focal_loss(
            pred_logits.flatten(0, 1),   # [bs*nq, T]
            tgt_full.flatten(0, 1),      # [bs*nq, T]
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return {"loss_ce": loss_ce}

    def _loss_boxes(
        self,
        outputs: dict,
        targets: List[dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> dict:
        pred_boxes = outputs["pred_boxes"]  # [bs, nq, 4] cxcywh normalized
        device = pred_boxes.device

        src_boxes_list = []
        tgt_boxes_list = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes_list.append(pred_boxes[i][src_idx])
            tgt_boxes_list.append(targets[i]["boxes"][tgt_idx].to(device))

        if len(src_boxes_list) == 0:
            zero = pred_boxes.sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = torch.cat(src_boxes_list, dim=0)  # [M, 4]
        tgt_boxes = torch.cat(tgt_boxes_list, dim=0)  # [M, 4]

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes

        # GIoU loss (diagonal of pairwise matrix = matched pairs)
        src_xyxy = _fix_degenerate_boxes(box_cxcywh_to_xyxy(src_boxes).clamp(0, 1))
        tgt_xyxy = _fix_degenerate_boxes(box_cxcywh_to_xyxy(tgt_boxes).clamp(0, 1))
        giou_matrix = generalized_box_iou(src_xyxy, tgt_xyxy)  # [M, M]
        loss_giou = (1 - torch.diag(giou_matrix)).sum() / num_boxes

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs: dict, targets: List[dict]) -> dict:
        """
        Args:
            outputs: dict with "pred_logits", "pred_boxes", optionally "aux_outputs".
            targets: list of target dicts (one per image in the batch).

        Returns:
            dict of scalar loss tensors keyed by loss name (+ suffix _i for aux layers).
        """
        # Run matcher on final layer outputs
        indices = self.matcher(outputs, targets)

        # Total matched GT boxes for normalization
        num_boxes = max(sum(len(t["boxes"]) for t in targets), 1)

        losses = {}
        losses.update(self._loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self._loss_boxes(outputs, targets, indices, num_boxes))

        # Auxiliary decoder layer losses
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_out, targets)
                for k, v in self._loss_labels(aux_out, targets, aux_indices, num_boxes).items():
                    losses[f"{k}_{i}"] = v
                for k, v in self._loss_boxes(aux_out, targets, aux_indices, num_boxes).items():
                    losses[f"{k}_{i}"] = v

        # CCM density loss (final layer only, not aux)
        if "pred_bbox_number" in outputs and outputs["pred_bbox_number"] is not None:
            ccm_targets = self._build_ccm_targets(targets, outputs["pred_bbox_number"].device)
            losses["loss_ccm"] = self.ccm_loss_fn(outputs["pred_bbox_number"], ccm_targets)

        return losses

    def _build_ccm_targets(self, targets: List[dict], device: torch.device) -> torch.Tensor:
        """Map actual object count to density class for AerialVG.

        Thresholds [3, 6, 10] → classes 0/1/2/3:
          0: 1-2 objects  → 200 queries
          1: 3-5 objects  → 300 queries
          2: 6-9 objects  → 450 queries
          3: 10+ objects  → 600 queries
        """
        thresholds = [3, 6, 10]
        labels = []
        for t in targets:
            n = len(t["boxes"])
            cls = sum(1 for thr in thresholds if n >= thr)
            labels.append(cls)
        return torch.tensor(labels, dtype=torch.long, device=device)
