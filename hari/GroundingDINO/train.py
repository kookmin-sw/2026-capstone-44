import argparse
import json
import os
import time
from pathlib import Path


import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.datasets.aerial_vg import build_aerial_vg_dataset, collate_fn
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.utils import sigmoid_focal_loss
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from groundingdino.util.get_tokenlizer import get_tokenlizer
from groundingdino.util.misc import get_world_size, is_dist_avail_and_initialized
from groundingdino.util.slconfig import SLConfig


# ──────────────────────────────────────────────────────────────────────────────
# Hungarian Matcher
# ──────────────────────────────────────────────────────────────────────────────

class HungarianMatcher(torch.nn.Module):
    """pred_logits / pred_boxes ↔ GT boxes / positive_map 매칭."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Returns:
            list of (src_indices, tgt_indices) tuples, one per image.
        """
        bs = outputs["pred_logits"].shape[0]
        indices = []

        for b in range(bs):
            out_logits = outputs["pred_logits"][b]   # [Q, max_text_len]
            out_bbox   = outputs["pred_boxes"][b]    # [Q, 4]
            tgt_boxes  = targets[b]["boxes"].to(out_bbox.device)        # [N, 4]
            tgt_pm     = targets[b]["positive_map"].to(out_logits.device)  # [N, max_text_len]

            num_tgt = tgt_boxes.shape[0]
            if num_tgt == 0:
                indices.append((
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                ))
                continue

            # ── Classification cost (focal) ───────────────────────────────
            alpha, gamma = self.focal_alpha, self.focal_gamma
            prob = out_logits.sigmoid()                        # [Q, T]
            prob_e = prob.unsqueeze(1)                         # [Q, 1, T]
            tgt_e  = tgt_pm.unsqueeze(0)                       # [1, N, T]

            neg_cost = (1 - alpha) * (prob_e ** gamma) * (-(1 - prob_e + 1e-8).log())
            pos_cost = alpha * ((1 - prob_e) ** gamma) * (-(prob_e + 1e-8).log())
            cost_class = (pos_cost * tgt_e - neg_cost * tgt_e).sum(-1)  # [Q, N]

            # ── L1 cost ──────────────────────────────────────────────────
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)  # [Q, N]

            # ── GIoU cost ────────────────────────────────────────────────
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_boxes),
            )  # [Q, N]

            C = (
                self.cost_class * cost_class
                + self.cost_bbox  * cost_bbox
                + self.cost_giou  * cost_giou
            )

            row_ind, col_ind = linear_sum_assignment(C.cpu().float().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices


# ──────────────────────────────────────────────────────────────────────────────
# Criterion
# ──────────────────────────────────────────────────────────────────────────────

class SetCriterion(torch.nn.Module):

    def __init__(
        self,
        matcher: HungarianMatcher,
        cls_loss_coef: float  = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        focal_alpha: float    = 0.25,
        focal_gamma: float    = 2.0,
    ):
        super().__init__()
        self.matcher       = matcher
        self.cls_coef      = cls_loss_coef
        self.bbox_coef     = bbox_loss_coef
        self.giou_coef     = giou_loss_coef
        self.focal_alpha   = focal_alpha
        self.focal_gamma   = focal_gamma

    def _compute_losses_for_output(self, out, targets, num_boxes):
        indices = self.matcher(out, targets)
        loss_cls = self._loss_labels(out, targets, indices, num_boxes)
        loss_bbox, loss_giou = self._loss_boxes(out, targets, indices, num_boxes)
        return loss_cls, loss_bbox, loss_giou

    def forward(self, outputs, targets):
        # ── normalize by total GT boxes across all ranks ──────────────────
        num_boxes = float(sum(t["boxes"].shape[0] for t in targets))
        num_boxes_t = torch.tensor([num_boxes], dtype=torch.float,
                                   device=outputs["pred_logits"].device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes_t)
        num_boxes = torch.clamp(num_boxes_t / get_world_size(), min=1.0).item()

        # ── main (final decoder layer) ────────────────────────────────────
        loss_cls, loss_bbox, loss_giou = self._compute_losses_for_output(outputs, targets, num_boxes)

        # ── aux losses (intermediate decoder layers) ──────────────────────
        aux_cls = aux_bbox = aux_giou = 0.0
        aux_outputs = outputs.get("aux_outputs", None)
        if aux_outputs is not None:
            for aux in aux_outputs:
                a_cls, a_bbox, a_giou = self._compute_losses_for_output(aux, targets, num_boxes)
                aux_cls  = aux_cls  + a_cls
                aux_bbox = aux_bbox + a_bbox
                aux_giou = aux_giou + a_giou

        # ── encoder two-stage intermediate output ──────────────────────────
        interm_outputs = outputs.get("interm_outputs", None)
        if interm_outputs is not None:
            i_cls, i_bbox, i_giou = self._compute_losses_for_output(interm_outputs, targets, num_boxes)
            aux_cls  = aux_cls  + i_cls
            aux_bbox = aux_bbox + i_bbox
            aux_giou = aux_giou + i_giou

        total_cls  = loss_cls  + aux_cls
        total_bbox = loss_bbox + aux_bbox
        total_giou = loss_giou + aux_giou

        total = self.cls_coef * total_cls + self.bbox_coef * total_bbox + self.giou_coef * total_giou

        return {
            "loss_total": total,
            "loss_cls":   total_cls,
            "loss_bbox":  total_bbox,
            "loss_giou":  total_giou,
        }

    # ── per-loss helpers ─────────────────────────────────────────────────────

    def _loss_labels(self, outputs, targets, indices, num_boxes):
        # clamp -inf (padding token mask) to avoid NaN from 0 * (-inf) in BCE
        src_logits = outputs["pred_logits"].clamp(min=-1e4)  # [B, Q, T]
        target_pm  = torch.zeros_like(src_logits)            # [B, Q, T]

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            pm = targets[b]["positive_map"].to(src_logits.device)  # [N, T]
            target_pm[b, src_idx] = pm[tgt_idx]

        return sigmoid_focal_loss(
            src_logits.flatten(0, 1),    # [B*Q, T]
            target_pm.flatten(0, 1),     # [B*Q, T]
            num_boxes=num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

    def _loss_boxes(self, outputs, targets, indices, num_boxes):
        device = outputs["pred_boxes"].device
        src_list, tgt_list = [], []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_list.append(outputs["pred_boxes"][b][src_idx])
            tgt_list.append(targets[b]["boxes"][tgt_idx].to(device))

        if not src_list:
            zero = outputs["pred_boxes"].sum() * 0.0   # keeps grad graph alive
            return zero, zero

        src = torch.cat(src_list)   # [M, 4]
        tgt = torch.cat(tgt_list)   # [M, 4]

        loss_bbox = F.l1_loss(src, tgt, reduction="sum") / num_boxes
        loss_giou = (
            1 - torch.diag(
                generalized_box_iou(box_cxcywh_to_xyxy(src), box_cxcywh_to_xyxy(tgt))
            )
        ).sum() / num_boxes

        return loss_bbox, loss_giou


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_device(targets, device):
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]


def reduce_dict(d: dict, device) -> dict:
    """average each value across DDP ranks."""
    out = {}
    for k, v in d.items():
        t = torch.tensor(v, dtype=torch.float64, device=device)
        dist.all_reduce(t)
        out[k] = (t / dist.get_world_size()).item()
    return out


def build_optimizer(model, cfg):
    # ── Freeze 설정 ──────────────────────────────────────────────────────────
    # freeze_mode 옵션 (yaml에서 조절):
    #   "none"     → 전부 학습 (기존 동작)
    #   "gate_only" → gate_w1, gate_w2만 학습, 나머지 전부 freeze
    #   "bert"     → bert만 freeze (기존 freeze_bert와 동일)
    freeze_mode = cfg.get("freeze_mode", "none")

    if freeze_mode == "gate_only":
        # (1) 전부 freeze
        for param in model.parameters():
            param.requires_grad_(False)
        # (2) gate + conv만 학습
        trainable_count = 0
        for name, param in model.named_parameters():
            if "gate_w" in name or "p3_down_convs" in name:
                param.requires_grad_(True)
                trainable_count += param.numel()
        # (3) gradient checkpointing 비활성화
        #     frozen 입력 → checkpoint가 grad graph을 끊어버리는 문제 방지
        base = model.module if hasattr(model, "module") else model
        for m in base.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = False
            if hasattr(m, "use_transformer_ckpt"):
                m.use_transformer_ckpt = False
        if dist.get_rank() == 0:
            total_count = sum(p.numel() for p in model.parameters())
            print(f"[freeze_mode=gate_only] trainable: {trainable_count:,} / {total_count:,} params")
            print(f"  gradient checkpointing disabled (frozen inputs)")
            # ── 실제 학습되는 파라미터 상세 출력 ──
            print("  trainable parameter breakdown:")
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(f"    {n}: shape={tuple(p.shape)}  numel={p.numel():,}")

    elif freeze_mode == "bbox_only":
        # bbox_embed만 학습, 나머지 전부 freeze
        # → decoder feature 불변 → logit ranking 보존 + box 정밀도만 개선
        for param in model.parameters():
            param.requires_grad_(False)
        trainable_count = 0
        for name, param in model.named_parameters():
            if "bbox_embed" in name:
                param.requires_grad_(True)
                trainable_count += param.numel()
        # gradient checkpointing 비활성화 (frozen inputs 문제 방지)
        base = model.module if hasattr(model, "module") else model
        for m in base.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = False
            if hasattr(m, "use_transformer_ckpt"):
                m.use_transformer_ckpt = False
        if dist.get_rank() == 0:
            total_count = sum(p.numel() for p in model.parameters())
            print(f"[freeze_mode=bbox_only] trainable: {trainable_count:,} / {total_count:,} params")

    elif freeze_mode == "bert":
        # BERT text encoder만 freeze (논문 baseline 설정과 동일)
        frozen_count = 0
        for name, param in model.named_parameters():
            if "bert" in name:
                param.requires_grad_(False)
                frozen_count += param.numel()
        if dist.get_rank() == 0:
            total_count = sum(p.numel() for p in model.parameters())
            print(f"[freeze_mode=bert] frozen: {frozen_count:,} / {total_count:,} params")
    # ── Optimizer param groups ────────────────────────────────────────────────
    lr_linear_proj_names = cfg.get("lr_linear_proj_names", [])  # e.g. ['ref_point_head', 'sampling_offsets']
    lr_linear_proj_mult  = cfg.get("lr_linear_proj_mult",  None)
    if isinstance(lr_linear_proj_mult, str):
        lr_linear_proj_mult = float(lr_linear_proj_mult)
    lr_fixed_names       = cfg.get("lr_fixed_names", ["gate_w", "p3_down_convs"])  # LR를 고정할 파라미터 이름 키워드

    bbox_embed_params, backbone_params, bert_params, linear_proj_params, rest_params, fixed_params = [], [], [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in lr_fixed_names):
            fixed_params.append(param)
        elif "bbox_embed" in name:
            bbox_embed_params.append(param)
        elif "backbone" in name:
            backbone_params.append(param)
        elif "bert" in name:
            bert_params.append(param)
        elif lr_linear_proj_mult is not None and any(kw in name for kw in lr_linear_proj_names):
            # ref_point_head, sampling_offsets: 위치 예측 관련 레이어 → 별도 저lr
            linear_proj_params.append(param)
        else:
            rest_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg.get("lr_backbone", 1e-5)},
        {"params": rest_params,     "lr": cfg.get("lr",          1e-4)},
    ]
    if bbox_embed_params:
        param_groups.append({"params": bbox_embed_params, "lr": cfg.get("lr_bbox_embed", cfg.get("lr", 1e-4))})
    if bert_params:
        param_groups.append({"params": bert_params, "lr": cfg.get("lr_bert", 1e-5)})
    if linear_proj_params and lr_linear_proj_mult is not None:
        param_groups.append({"params": linear_proj_params, "lr": lr_linear_proj_mult})
    if fixed_params:
        # fixed_lr: True 태그를 달아두면 scheduler에서 이 그룹만 LR을 고정
        param_groups.append({"params": fixed_params, "lr": cfg.get("lr_fixed", 1e-4), "fixed_lr": True})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))


# ──────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, epoch, is_main, output_dir=None):
    model.train()
    acc = {k: 0.0 for k in ("loss_total", "loss_cls", "loss_bbox", "loss_giou")}
    clip = cfg.get("clip_max_norm", 0.1)
    save_every = cfg.get("save_every_steps", 500)
    t0 = time.time()

    total_steps = len(loader)
    log_every = cfg.get("log_every_steps", 50)
    for step, (images, targets) in enumerate(loader):
        images  = images.to(device)
        targets = to_device(targets, device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, targets)
            losses  = criterion(outputs, targets)

        scaler.scale(losses["loss_total"]).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        for k in acc:
            acc[k] += losses[k].item()

        if is_main and (step + 1) % log_every == 0:
            print(
                f"  [train] epoch {epoch+1} step {step+1}/{total_steps}  "
                f"loss={losses['loss_total'].item():.4f}  "
                f"cls={losses['loss_cls'].item():.4f}  "
                f"bbox={losses['loss_bbox'].item():.4f}  "
                f"giou={losses['loss_giou'].item():.4f}",
                flush=True,
            )

        if is_main and output_dir and save_every > 0 and (step + 1) % save_every == 0:
            ckpt_path = Path(output_dir) / f"checkpoint_step_latest.pth"
            # 이전 step 체크포인트 삭제 후 저장
            if ckpt_path.exists():
                ckpt_path.unlink()
            torch.save({
                "epoch": epoch,
                "step": step + 1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}", flush=True)

    avg = {k: v / len(loader) for k, v in acc.items()}
    if is_main:
        print(
            f"  [train] epoch {epoch+1} done  "
            f"loss={avg['loss_total']:.4f}  "
            f"cls={avg['loss_cls']:.4f}  "
            f"bbox={avg['loss_bbox']:.4f}  "
            f"giou={avg['loss_giou']:.4f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    return avg


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    acc = {k: 0.0 for k in ("loss_total", "loss_cls", "loss_bbox", "loss_giou")}

    for images, targets in loader:
        images  = images.to(device)
        targets = to_device(targets, device)
        with torch.cuda.amp.autocast():
            outputs = model(images, targets)
            losses  = criterion(outputs, targets)
        for k in acc:
            acc[k] += losses[k].item()

    return {k: v / max(len(loader), 1) for k, v in acc.items()}


@torch.no_grad()
def compute_accuracy(model, cfg, tokenizer, device):
    """Top-1/Top-5 Acc@0.5 — distributed: val set을 rank끼리 나눠서 처리 후 all_reduce."""
    import os
    from torch.utils.data.distributed import DistributedSampler
    from groundingdino.datasets.aerial_vg import AerialVGEvalDataset, eval_collate_fn

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    annotation_file = os.path.join(cfg["annotation_dir"], "vg_test_odvg.jsonl")
    ds = AerialVGEvalDataset(cfg["image_dir"], annotation_file)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, sampler=sampler,
                        num_workers=4, collate_fn=eval_collate_fn)

    model.eval()
    top1_correct = torch.tensor(0, dtype=torch.long, device=device)
    top5_correct = torch.tensor(0, dtype=torch.long, device=device)
    total        = torch.tensor(0, dtype=torch.long, device=device)

    for images, captions, gt_boxes in loader:
        images   = images.to(device)
        gt_boxes = gt_boxes.to(device)

        outputs = model(images, captions=captions)
        logits = outputs["pred_logits"].sigmoid().max(dim=2)[0]  # [B, Q]
        boxes  = outputs["pred_boxes"]                            # [B, Q, 4]

        _, top_indices = logits.topk(min(5, logits.shape[1]), dim=1)
        boxes_sel = boxes.gather(1, top_indices.unsqueeze(-1).expand(-1, -1, 4))

        # cxcywh → xyxy for IoU
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes).unsqueeze(1)  # [B, 1, 4]
        cd_xyxy = box_cxcywh_to_xyxy(boxes_sel)              # [B, K, 4]

        inter_x1 = torch.max(gt_xyxy[..., 0], cd_xyxy[..., 0])
        inter_y1 = torch.max(gt_xyxy[..., 1], cd_xyxy[..., 1])
        inter_x2 = torch.min(gt_xyxy[..., 2], cd_xyxy[..., 2])
        inter_y2 = torch.min(gt_xyxy[..., 3], cd_xyxy[..., 3])

        inter    = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        gt_area  = (gt_xyxy[..., 2] - gt_xyxy[..., 0]) * (gt_xyxy[..., 3] - gt_xyxy[..., 1])
        cd_area  = (cd_xyxy[..., 2] - cd_xyxy[..., 0]) * (cd_xyxy[..., 3] - cd_xyxy[..., 1])
        union    = gt_area + cd_area - inter
        iou      = inter / (union + 1e-6)  # [B, K]

        top1_correct += (iou[:, 0] > 0.5).sum()
        top5_correct += (iou.max(dim=1).values > 0.5).sum()
        total        += iou.shape[0]

    # 모든 rank의 결과를 합산
    dist.all_reduce(top1_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(top5_correct, op=dist.ReduceOp.SUM)

    # padding 샘플 제외: 실제 dataset 크기를 분모로 사용
    n = len(ds)
    return {
        "top1_acc": top1_correct.item() / n * 100,
        "top5_acc": top5_correct.item() / n * 100,
        "total":    n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── DDP init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    output_dir = Path(cfg["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"World size: {world_size}  |  output: {output_dir}")

    # ── Logging ──────────────────────────────────────────────────────────────
    if is_main and cfg.get("use_wandb", False):
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "groundingdino"),
            name=cfg.get("run_name", "aerial_vg"),
            config=cfg,
        )

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = SLConfig.fromfile(cfg["config_file"])
    model     = build_model(model_cfg)

    if cfg.get("pretrained_checkpoint"):
        ckpt       = torch.load(cfg["pretrained_checkpoint"], map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        # strip DDP 'module.' prefix if present
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main:
            print(f"Loaded pretrained: {cfg['pretrained_checkpoint']}")
            print(f"missing keys: {missing}")
            print(f"unexpected keys: {unexpected}")

    model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model._set_static_graph()

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    tokenizer = get_tokenlizer(model_cfg.text_encoder_type)

    def make_loader(split, shuffle):
        ds = build_aerial_vg_dataset(
            image_dir      = cfg["image_dir"],
            annotation_dir = cfg["annotation_dir"],
            tokenizer      = tokenizer,
            image_set      = split,
            max_text_len   = cfg.get("max_text_len", 256),
        )
        sampler = DistributedSampler(ds, shuffle=shuffle)
        return DataLoader(
            ds,
            batch_size  = cfg["batch_size"],
            sampler     = sampler,
            num_workers = cfg.get("num_workers", 4),
            collate_fn  = collate_fn,
            pin_memory  = True,
            drop_last   = (split == "train"),
        ), sampler

    train_loader, train_sampler = make_loader("train", shuffle=True)
    val_loader,   _             = make_loader("val",   shuffle=False)

    if is_main:
        print(f"Train: {len(train_loader.dataset)} samples  |  Val: {len(val_loader.dataset)} samples")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer    = build_optimizer(model, cfg)
    lr_drop_list = cfg.get("lr_drop_list", None)
    if lr_drop_list:
        milestones = lr_drop_list
        def _make_scheduled_fn(ms):
            def fn(epoch):
                decay = 1.0
                for m in ms:
                    if epoch >= m:
                        decay *= 0.1
                return decay
            return fn
        scheduled_fn = _make_scheduled_fn(milestones)
    else:
        lr_drop = cfg.get("lr_drop", 10)
        scheduled_fn = lambda epoch: 0.1 ** (epoch // lr_drop)

    # fixed_lr 태그가 달린 그룹(추가 모듈)은 항상 1.0 (LR 고정), 나머지는 decay 적용
    lr_lambdas = [
        (lambda epoch: 1.0) if pg.get("fixed_lr", False) else scheduled_fn
        for pg in optimizer.param_groups
    ]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
    scaler = torch.cuda.amp.GradScaler()

    # ── Criterion ────────────────────────────────────────────────────────────
    matcher   = HungarianMatcher(
        cost_class  = cfg.get("set_cost_class", 1.0),
        cost_bbox   = cfg.get("set_cost_bbox",  5.0),
        cost_giou   = cfg.get("set_cost_giou",  2.0),
        focal_alpha = cfg.get("focal_alpha",    0.25),
        focal_gamma = cfg.get("focal_gamma",    2.0),
    )
    criterion = SetCriterion(
        matcher        = matcher,
        cls_loss_coef  = cfg.get("cls_loss_coef",  1.0),
        bbox_loss_coef = cfg.get("bbox_loss_coef", 5.0),
        giou_loss_coef = cfg.get("giou_loss_coef", 2.0),
        focal_alpha    = cfg.get("focal_alpha",    0.25),
        focal_gamma    = cfg.get("focal_gamma",    2.0),
    ).to(device)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.module.load_state_dict(ckpt["model"], strict=False)
        # strict=False: 새로 추가된 모듈(gate_w1, gate_w2 등)은 랜덤 초기화 상태로 유지
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        except ValueError:
            # param groups 구성이 달라진 경우(예: freeze_mode 변경) optimizer는 새로 시작
            if is_main:
                print("  optimizer param groups mismatch → optimizer/scheduler reset (fresh start)")
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}  best_val_loss={best_val_loss:.4f}")
            if missing:
                print(f"  missing keys (new modules, train from scratch): {len(missing)} keys")

    # ── Training loop ────────────────────────────────────────────────────────
    epochs = cfg.get("epochs", 12)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)

        if is_main:
            print(f"\n{'='*60}")
            # param_groups[1]이 rest_params (main lr). 0=backbone
            main_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}  main_lr={main_lr:.2e}")

        # Train
        train_losses = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, device, cfg, epoch, is_main,
            output_dir=output_dir,
        )
        train_losses = reduce_dict(train_losses, device)

        lr_scheduler.step()

        # Val
        val_losses = evaluate(model, criterion, val_loader, device)
        val_losses = reduce_dict(val_losses, device)

        # Accuracy — acc_eval_freq 에폭마다 한 번만 계산
        acc_eval_freq = cfg.get("acc_eval_freq", 1)
        run_acc = (epoch + 1) % acc_eval_freq == 0 or (epoch + 1) == epochs
        acc_result = compute_accuracy(model.module, cfg, tokenizer, device) if run_acc else None

        # Logging + checkpointing (rank 0 only)
        if is_main:
            print(
                f"  train loss={train_losses['loss_total']:.4f} "
                f"(cls={train_losses['loss_cls']:.4f} "
                f"bbox={train_losses['loss_bbox']:.4f} "
                f"giou={train_losses['loss_giou']:.4f})"
            )
            print(
                f"  val   loss={val_losses['loss_total']:.4f} "
                f"(cls={val_losses['loss_cls']:.4f} "
                f"bbox={val_losses['loss_bbox']:.4f} "
                f"giou={val_losses['loss_giou']:.4f})"
            )
            if acc_result is not None:
                print(
                    f"  Top-1 Accuracy={acc_result['top1_acc']:.2f}%  "
                    f"Top-5 Accuracy={acc_result['top5_acc']:.2f}%  "
                    f"(phrases={acc_result['total']})"
                )

            if cfg.get("use_wandb", False):
                import wandb
                log_dict = {
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_losses.items()},
                    **{f"val/{k}":   v for k, v in val_losses.items()},
                    "lr": optimizer.param_groups[-1]["lr"],
                }
                if acc_result is not None:
                    log_dict["val/top1_acc"] = acc_result["top1_acc"]
                    log_dict["val/top5_acc"] = acc_result["top5_acc"]
                wandb.log(log_dict)

            loss_log_path = output_dir / "loss_log.jsonl"
            entry = {
                "epoch": epoch + 1,
                "train": train_losses,
                "val":   val_losses,
            }
            if acc_result is not None:
                entry["top1_acc"] = acc_result["top1_acc"]
                entry["top5_acc"] = acc_result["top5_acc"]
            with loss_log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

            ckpt = {
                "epoch":        epoch,
                "model":        model.module.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "val_loss":     val_losses["loss_total"],
                "cfg":          cfg,
            }

            if val_losses["loss_total"] < best_val_loss:
                best_val_loss = val_losses["loss_total"]
                torch.save(ckpt, output_dir / "checkpoint_best.pth")
                print(f"  *** New best val loss: {best_val_loss:.4f} → saved checkpoint_best.pth")

            if (epoch + 1) % cfg.get("save_freq", 1) == 0:
                torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch+1:04d}.pth")

        dist.barrier()

    dist.destroy_process_group()
    if is_main and cfg.get("use_wandb", False):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()