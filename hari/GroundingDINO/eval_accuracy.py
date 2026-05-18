import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from groundingdino.datasets.aerial_vg import AerialVGEvalDataset, eval_collate_fn
from groundingdino.models import build_model
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
from groundingdino.util.slconfig import SLConfig


def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def disable_stochastic(model):
    """DropPath / StochasticDepth 등 eval() 로 꺼지지 않는 레이어 강제 비활성화."""
    for module in model.modules():
        if hasattr(module, "drop_prob"):
            module.drop_prob = 0.0
        if hasattr(module, "drop_path_rate"):
            module.drop_path_rate = 0.0


@torch.no_grad()
def evaluate(model, cfg, device, split="val", seed: int = 42):
    # ── 결정론적 실행 보장 ─────────────────────────────────────────
    set_deterministic(seed)
    disable_stochastic(model)

    annotation_file = os.path.join(cfg["annotation_dir"], f"vg_{split}_odvg.jsonl")
    ds = AerialVGEvalDataset(cfg["image_dir"], annotation_file)

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=0, collate_fn=eval_collate_fn,
                        worker_init_fn=seed_worker, generator=g)

    model.eval()
    top1_correct = 0
    top5_correct = 0
    total_seen   = 0

    for i, (images, captions, gt_boxes) in enumerate(loader):
        images   = images.to(device)
        gt_boxes = gt_boxes.to(device)
        B        = len(captions)
        total_seen += B

        # ── 진단: 첫 배치 ──────────────────────────────────────────
        if i == 0:
            print(f"\n[Diag/{split}] gt_boxes range: "
                  f"min={gt_boxes.min():.4f} max={gt_boxes.max():.4f}")
            token_lengths = [len(c.split()) for c in captions]
            print(f"[Diag/{split}] caption token len: "
                  f"min={min(token_lengths)} max={max(token_lengths)} "
                  f"mean={sum(token_lengths)/len(token_lengths):.1f}")

        outputs = model(images, captions=captions)
        logits_raw = outputs["pred_logits"].sigmoid()     # [B, Q, T]
        boxes      = outputs["pred_boxes"]                # [B, Q, 4]

        # ── 패딩 토큰 제거: caption별 실제 토큰 수만 사용 ──────────
        token_mask    = (logits_raw.max(dim=1).values > 0.0)  # [B, T]
        logits_masked = logits_raw * token_mask.unsqueeze(1)  # [B, Q, T]
        query_scores  = logits_masked.max(dim=2).values       # [B, Q]

        K = min(5, query_scores.shape[1])
        _, top_indices = query_scores.topk(K, dim=1)
        boxes_sel = boxes.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, 4)
        )

        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes).unsqueeze(1)  # [B, 1, 4]
        cd_xyxy = box_cxcywh_to_xyxy(boxes_sel)              # [B, K, 4]

        inter_x1 = torch.max(gt_xyxy[..., 0], cd_xyxy[..., 0])
        inter_y1 = torch.max(gt_xyxy[..., 1], cd_xyxy[..., 1])
        inter_x2 = torch.min(gt_xyxy[..., 2], cd_xyxy[..., 2])
        inter_y2 = torch.min(gt_xyxy[..., 3], cd_xyxy[..., 3])

        inter   = (torch.clamp(inter_x2 - inter_x1, min=0)
                   * torch.clamp(inter_y2 - inter_y1, min=0))
        gt_area = ((gt_xyxy[..., 2] - gt_xyxy[..., 0])
                   * (gt_xyxy[..., 3] - gt_xyxy[..., 1]))
        cd_area = ((cd_xyxy[..., 2] - cd_xyxy[..., 0])
                   * (cd_xyxy[..., 3] - cd_xyxy[..., 1]))
        union   = gt_area + cd_area - inter
        iou     = inter / (union + 1e-6)  # [B, K]

        top1_correct += (iou[:, 0] >= 0.5).sum().item()
        top5_correct += (iou.max(dim=1).values >= 0.5).sum().item()

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(loader)}] "
                  f"Top-1={top1_correct/total_seen*100:.2f}%  "
                  f"Top-5={top5_correct/total_seen*100:.2f}%")

    n = len(ds)
    print(f"\nSplit         : {split}")
    print(f"Total phrases : {n}")
    print(f"Top-1 Accuracy: {top1_correct/n*100:.2f}%  ({top1_correct}/{n})")
    print(f"Top-5 Accuracy: {top5_correct/n*100:.2f}%  ({top5_correct}/{n})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--seed",       default=42, type=int)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = SLConfig.fromfile(cfg["config_file"])
    model_cfg.use_p3_skip = True 
    model = build_model(model_cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"Loaded: {args.checkpoint}")
    print(f"missing keys: {missing}")
    print(f"unexpected keys: {unexpected}")

    evaluate(model, cfg, device, split=args.split, seed=args.seed)

if __name__ == "__main__":
    main()