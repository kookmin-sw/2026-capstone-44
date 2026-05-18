"""
Re-verify zero-shot GDINO (vanilla weights) on AerialVG val.jsonl.
Same protocol as AerialVG paper eval.py:
  - WEIGHTS: groundingdino_swint_ogc.pth (vanilla, NO AerialVG training)
  - Caption: original grounding.caption (natural language)
  - Score: pred_logits.sigmoid().max(-1).values
  - Top-1: best-scored box IoU > 0.5
  - Top-5: any of top-5 boxes IoU > 0.5
"""
import sys, os, json, math, argparse, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from groundingdino.util.inference import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
import groundingdino.datasets.transforms as T

CONFIG    = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS   = "./checkpoints/groundingdino_swint_ogc.pth"
VAL_ANNO  = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT  = "/data2/huggingface/AerialVG/images"
DEVICE    = "cuda:0"
TOPK      = 5

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
        return image_t, meta["grounding"]["caption"], gt_norm

def collate_fn(batch):
    imgs, caps, gts = zip(*batch)
    return list(imgs), list(caps), torch.stack(gts)

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
    os.chdir(Path(__file__).parent)
    print(f"WEIGHTS: {WEIGHTS}  (vanilla GDINO, NO fine-tuning)")
    print(f"ANNO: {VAL_ANNO}")
    model = load_model(CONFIG, WEIGHTS)
    model = model.to(DEVICE).eval()

    ds = AerialVGEvalDataset(VAL_ANNO, IMG_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Val samples: {len(ds)}")

    top1, top5, total = 0, 0, 0
    for images, captions, gt_boxes in tqdm(loader, desc="Eval", leave=False):
        try:
            img_batch = nested_tensor_from_tensor_list(images).to(DEVICE)
            gt = gt_boxes[0].to(DEVICE)
            out = model(img_batch, captions=captions)
            scores = out["pred_logits"].sigmoid().max(dim=-1).values[0]
            boxes  = cxcywh_to_xyxy(out["pred_boxes"][0])
            top_idx = scores.topk(TOPK).indices
            top_boxes = boxes[top_idx]

            iou1 = box_iou(top_boxes[0].tolist(), gt.tolist())
            top1 += float(iou1 > 0.5)
            hit5 = any(box_iou(top_boxes[i].tolist(), gt.tolist()) > 0.5 for i in range(TOPK))
            top5 += float(hit5)
            total += 1
        except RuntimeError as e:
            print(f"skip: {e}")

    print(f"\n{'='*50}")
    print(f"  Vanilla GDINO (groundingdino_swint_ogc.pth)")
    print(f"  Top-1 Acc@0.5: {top1/total:.4f} ({top1/total*100:.2f}%)")
    print(f"  Top-5 Acc@0.5: {top5/total:.4f} ({top5/total*100:.2f}%)")
    print(f"  Total: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
