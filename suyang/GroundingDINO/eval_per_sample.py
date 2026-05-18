"""
Per-sample evaluation: save Top-1 / Top-5 IoU + metadata for each val sample.
Output: per_sample_eval.jsonl (one record per sample)
"""
import sys, os, json, math, argparse, torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
import groundingdino.datasets.transforms as T
from groundingdino.models.GroundingDINO.relation_v3 import SpatialRelationBiasModule

# ── LoRA wrappers ─────────────────────────────────────────────────────────────
class QKLoRAQKV(nn.Module):
    def __init__(self, qkv_linear, rank=8, alpha=16.0):
        super().__init__()
        self.qkv = qkv_linear
        d = qkv_linear.in_features
        self.scale = alpha / rank
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d))
        self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
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
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
        self.lora_k_A = nn.Parameter(torch.empty(rank, d))
        self.lora_k_B = nn.Parameter(torch.zeros(d, rank))
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
        self.lora_q_A = nn.Parameter(torch.empty(rank, d))
        self.lora_q_B = nn.Parameter(torch.zeros(d, rank))
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
TOPK      = 15

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
        return image_t, meta["grounding"]["caption"], gt_norm, meta, idx

def collate_fn(batch):
    imgs, caps, gts, metas, idxs = zip(*batch)
    return list(imgs), list(caps), torch.stack(gts), list(metas), list(idxs)

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--srbm_layers",type=int, default=3)
    parser.add_argument("--out",        default="per_sample_eval.jsonl")
    parser.add_argument("--device",     default=DEVICE)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    model = load_model(CONFIG, BACKBONE)
    model.relation_weight = 0.0
    apply_all_lora(model, rank=8, alpha=16.0)

    lora_ckpt = torch.load(LORA_CKPT, map_location="cpu")
    model.load_state_dict(lora_ckpt["model"], strict=False)

    model.relation_transformer = SpatialRelationBiasModule(
        d_model=256, num_heads=8, num_layers=args.srbm_layers,
        topk=TOPK, max_text_len=256,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(args.device).eval()
    srbm = model.relation_transformer

    ds = AerialVGEvalDataset(VAL_ANNO, IMG_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Val samples: {len(ds)}")

    records = []
    for images, captions, gt_boxes, metas, idxs in tqdm(loader, desc="Eval"):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(args.device)
            gt = gt_boxes[0].to(args.device)
            meta = metas[0]; idx = idxs[0]

            model.relation_weight = 0.0
            model(img_batch, captions=captions)
            cache = model._cache

            final_hs = cache["final_hs"].detach()
            final_pred_boxes = cache["final_pred_boxes"].detach()
            final_pred_logits = cache["final_pred_logits"].detach()
            text_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in cache["text_dict"].items()}

            scores = final_pred_logits.sigmoid().max(dim=-1).values
            topk_idx = scores.topk(TOPK, dim=1).indices
            topk_feats = final_hs.gather(1, topk_idx.unsqueeze(-1).expand(-1,-1,final_hs.size(-1)))
            topk_coords = final_pred_boxes.gather(1, topk_idx.unsqueeze(-1).expand(-1,-1,4))

            srbm_logits = srbm(topk_feats, topk_coords, text_dict)
            srbm_scores = srbm_logits.max(dim=-1).values[0]

            ranked = srbm_scores.argsort(descending=True)
            ranked_boxes_xyxy = cxcywh_to_xyxy(topk_coords[0])[ranked]

            ious = [box_iou(ranked_boxes_xyxy[i].tolist(), gt.tolist()) for i in range(15)]
            top1_iou = ious[0]
            top5_iou = max(ious[:5])
            top15_iou = max(ious)

            # Metadata
            regions = meta["grounding"]["regions"]
            anchor = regions[0]
            ab = anchor["bbox"]
            anchor_w = (ab[2] - ab[0]) / meta["width"]
            anchor_h = (ab[3] - ab[1]) / meta["height"]
            anchor_area = anchor_w * anchor_h
            refs = [r for r in regions[1:] if "realation" in r]
            relations = [r["realation"] for r in refs]

            records.append({
                "idx": idx,
                "filename": meta["filename"],
                "anchor_phrase": anchor.get("phrase", ""),
                "anchor_area": float(anchor_area),
                "anchor_w": float(anchor_w),
                "anchor_h": float(anchor_h),
                "num_refs": len(refs),
                "relations": relations,
                "top1_iou": float(top1_iou),
                "top5_iou": float(top5_iou),
                "top15_iou": float(top15_iou),
                "top1_correct": int(top1_iou > 0.5),
                "top5_correct": int(top5_iou > 0.5),
                "caption": meta["grounding"]["caption"],
            })
        except RuntimeError as e:
            print(f"skip idx {idxs[0]}: {e}")

    out_path = args.out
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    top1 = sum(r["top1_correct"] for r in records) / len(records)
    top5 = sum(r["top5_correct"] for r in records) / len(records)
    print(f"\nTotal {len(records)}  Top-1 {top1*100:.2f}%  Top-5 {top5*100:.2f}%")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
