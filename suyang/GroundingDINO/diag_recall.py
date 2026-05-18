"""
Diagnostic: measure recall@K for K = 1, 5, 15, 30, 50, 100, 900 (all queries).
Tests GDINO+LoRA base detector (no SRBM) — just checks where GT appears in rank list.
"""
import sys, os, json, math, torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
import groundingdino.datasets.transforms as T

# ── LoRA wrappers (same as eval_per_sample) ─────────────────────────────────────
class QKLoRAQKV(nn.Module):
    def __init__(self, qkv_linear, rank=8, alpha=16.0):
        super().__init__()
        self.qkv = qkv_linear; d = qkv_linear.in_features; self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d)); self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d)); self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
        for p in self.qkv.parameters(): p.requires_grad = False
    def forward(self, x):
        qkv = self.qkv(x); d = x.shape[-1]
        qkv[..., :d]    += (x @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        qkv[..., d:2*d] += (x @ self.lora_k_A.T @ self.lora_k_B.T) * self.scale
        return qkv

class QKLoRAMHA(nn.Module):
    def __init__(self, mha, rank=8, alpha=16.0):
        super().__init__()
        self.mha = mha; d = mha.embed_dim; self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d)); self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d)); self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
        for p in self.mha.parameters(): p.requires_grad = False
    def forward(self, query, key, value, **kw):
        query = query + (query @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        key   = key   + (key   @ self.lora_k_A.T @ self.lora_k_B.T) * self.scale
        return self.mha(query, key, value, **kw)

class QLoRAMSDeformAttn(nn.Module):
    def __init__(self, da, rank=8, alpha=16.0):
        super().__init__()
        self.deform_attn = da; d = da.embed_dim; self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d)); self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        for p in self.deform_attn.parameters(): p.requires_grad = False
    def forward(self, query, reference_points, value, spatial_shapes,
                level_start_index=None, key_padding_mask=None, **kw):
        query = query + (query @ self.lora_q_A.T @ self.lora_q_B.T) * self.scale
        return self.deform_attn(query, reference_points, value, spatial_shapes,
                                level_start_index=level_start_index,
                                key_padding_mask=key_padding_mask, **kw)

def apply_all_lora(model, rank=8, alpha=16.0):
    for module in model.modules():
        if type(module).__name__ == "WindowAttention":
            if isinstance(module.qkv, nn.Linear):
                module.qkv = QKLoRAQKV(module.qkv, rank, alpha)
        if hasattr(module, "self_attn") and isinstance(module.self_attn, nn.MultiheadAttention):
            module.self_attn = QKLoRAMHA(module.self_attn, rank, alpha)
        if hasattr(module, "ca_text") and isinstance(module.ca_text, nn.MultiheadAttention):
            module.ca_text = QKLoRAMHA(module.ca_text, rank, alpha)
        if hasattr(module, "cross_attn") and type(module.cross_attn).__name__ == "MSDeformAttn":
            module.cross_attn = QLoRAMSDeformAttn(module.cross_attn, rank, alpha)

CONFIG    = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
BACKBONE  = "./checkpoints/groundingdino_swint_ogc.pth"
LORA_CKPT = "./output/lora_only/best.pth"
VAL_ANNO  = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT  = "/data2/huggingface/AerialVG/images"
DEVICE    = "cuda:0"

TRANSFORM = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class AerialVGEvalDataset(Dataset):
    def __init__(self, anno_path, img_root):
        with open(anno_path) as f:
            self.metas = [json.loads(l) for l in f if l.strip()]
        self.img_root = img_root
    def __len__(self): return len(self.metas)
    def __getitem__(self, idx):
        meta = self.metas[idx]
        image = Image.open(os.path.join(self.img_root, meta["filename"])).convert("RGB")
        w, h = image.size
        image_t, _ = TRANSFORM(image, None)
        anchor = meta["grounding"]["regions"][0]
        gt_bbox = anchor["bbox"]
        gt_norm = torch.tensor([gt_bbox[0]/w, gt_bbox[1]/h, gt_bbox[2]/w, gt_bbox[3]/h], dtype=torch.float32)
        return image_t, meta["grounding"]["caption"], gt_norm, meta["filename"]

def collate_fn(batch):
    imgs, caps, gts, fns = zip(*batch)
    return list(imgs), list(caps), torch.stack(gts), list(fns)

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou_vec(boxes, gt):
    # boxes [N,4] xyxy, gt [4] xyxy
    x1 = torch.maximum(boxes[:,0], gt[0])
    y1 = torch.maximum(boxes[:,1], gt[1])
    x2 = torch.minimum(boxes[:,2], gt[2])
    y2 = torch.minimum(boxes[:,3], gt[3])
    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    area_b = (boxes[:,2]-boxes[:,0]).clamp(0) * (boxes[:,3]-boxes[:,1]).clamp(0)
    area_g = (gt[2]-gt[0]).clamp(0) * (gt[3]-gt[1]).clamp(0)
    return inter / (area_b + area_g - inter + 1e-6)

@torch.no_grad()
def main():
    os.chdir(Path(__file__).parent)
    print(f"Loading vanilla GDINO + LoRA...")
    model = load_model(CONFIG, BACKBONE)
    apply_all_lora(model, rank=8, alpha=16.0)
    lora_ckpt = torch.load(LORA_CKPT, map_location="cpu")
    model.load_state_dict(lora_ckpt["model"], strict=False)
    model = model.to(DEVICE).eval()

    ds = AerialVGEvalDataset(VAL_ANNO, IMG_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    Ks = [1, 5, 15, 30, 50, 100, 900]
    recalls = {k: 0 for k in Ks}
    gt_ranks = []  # for each sample: rank position where GT first appears with IoU>0.5 (-1 if never)
    total = 0

    for images, captions, gt_boxes, fns in tqdm(loader, desc="Diag"):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
            gt = gt_boxes[0].to(DEVICE)
            out = model(img_batch, captions=captions)
            scores = out["pred_logits"].sigmoid().max(dim=-1).values[0]  # [900]
            boxes_xyxy = cxcywh_to_xyxy(out["pred_boxes"][0])              # [900,4]

            order = scores.argsort(descending=True)
            ranked = boxes_xyxy[order]                                     # [900,4] sorted by score
            ious = box_iou_vec(ranked, gt)                                 # [900]
            hits = (ious > 0.5)

            # First rank where IoU > 0.5
            if hits.any():
                first_hit = hits.nonzero()[0].item() + 1  # 1-indexed
            else:
                first_hit = -1
            gt_ranks.append(first_hit)

            for k in Ks:
                if hits[:k].any():
                    recalls[k] += 1
            total += 1
        except RuntimeError as e:
            print(f"skip {fns[0]}: {e}")

    print(f"\nTotal: {total}")
    print(f"\n=== Recall @ K ===")
    print(f"{'K':>6}  {'Recall':>8}  {'Δ vs prev':>10}")
    prev = 0
    for k in Ks:
        r = recalls[k] / total * 100
        delta = r - prev
        print(f"{k:>6}  {r:>7.2f}%  {delta:>9.2f}%p")
        prev = r

    # Distribution of first_hit
    import numpy as np
    arr = np.array(gt_ranks)
    miss = (arr == -1).sum()
    hit = (arr > 0).sum()
    print(f"\n=== First-hit rank distribution ===")
    print(f"Never hit (all 900 miss): {miss}  ({miss/total*100:.1f}%)")
    print(f"Hit somewhere:            {hit}  ({hit/total*100:.1f}%)")
    if hit > 0:
        hit_ranks = arr[arr > 0]
        print(f"  median rank: {int(np.median(hit_ranks))}")
        print(f"  mean rank:   {hit_ranks.mean():.1f}")
        print(f"  p25 / p75:   {int(np.percentile(hit_ranks, 25))} / {int(np.percentile(hit_ranks, 75))}")

    # save raw ranks
    with open("diag_ranks.json", "w") as f:
        json.dump({"total": total, "recalls": {str(k):v for k,v in recalls.items()},
                   "first_hit_ranks": gt_ranks}, f)
    print("\nSaved: diag_ranks.json")


if __name__ == "__main__":
    main()
