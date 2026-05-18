import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image, Image as PILImage
from torchvision.ops import box_iou, generalized_box_iou
from tqdm import tqdm

from groundingdino.models import build_groundingdino
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict


# ==========================================================
# Dataset
# ==========================================================
class AerialVGDataset(Dataset):
    def __init__(self, hf_dataset, image_dir, transform=None):
        self.dataset = hf_dataset
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img_path = os.path.join(self.image_dir, item["filename"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = item["width"], item["height"]

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        text = item["grounding"]["caption"]

        boxes = []
        for region in item["grounding"]["regions"]:
            x_min, y_min, x_max, y_max = region["bbox"]
            cx = ((x_min + x_max) / 2.0) / orig_w
            cy = ((y_min + y_max) / 2.0) / orig_h
            w = (x_max - x_min) / orig_w
            h = (y_max - y_min) / orig_h
            boxes.append([cx, cy, w, h])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "caption": text,
        }
        return image_tensor, target, text


def custom_collate_fn(batch):
    images  = nested_tensor_from_tensor_list([item[0] for item in batch])
    targets = [item[1] for item in batch]
    texts   = [item[2] for item in batch]
    return images, targets, texts


# ==========================================================
# Utils
# ==========================================================
def cxcywh_to_xyxy(boxes):
    return torch.stack([
        boxes[:, 0] - boxes[:, 2] / 2,
        boxes[:, 1] - boxes[:, 3] / 2,
        boxes[:, 0] + boxes[:, 2] / 2,
        boxes[:, 1] + boxes[:, 3] / 2,
    ], dim=1)


def ranking_loss(pred_boxes, pred_logits, targets, w_l1=2.0, w_giou=1.0, w_bce=0.5, iou_pos_thr=0.3):
    """
    1) CrossEntropy: GT와 IoU 최대인 query를 Top-1으로 ranking
    2) L1 + GIoU: 그 query의 box를 GT 쪽으로 regression
    3) BCE: IoU > iou_pos_thr인 query 여러 개를 동시에 positive → Top-5 개선
    """
    B = pred_boxes.shape[0]
    total_cls  = torch.tensor(0.0, device=pred_boxes.device)
    total_l1   = torch.tensor(0.0, device=pred_boxes.device)
    total_giou = torch.tensor(0.0, device=pred_boxes.device)
    total_bce  = torch.tensor(0.0, device=pred_boxes.device)
    n_valid = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"]          # [M, 4] cxcywh
        if gt_boxes.shape[0] == 0:
            continue

        pb_xyxy = cxcywh_to_xyxy(pred_boxes[b])   # [N, 4]
        gb_xyxy = cxcywh_to_xyxy(gt_boxes)         # [M, 4]

        iou = box_iou(pb_xyxy, gb_xyxy)            # [N, M]
        best_iou_per_query = iou.max(dim=1)[0]     # [N]
        best_query_idx = best_iou_per_query.argmax()  # scalar

        # GT box: best query와 IoU 최대인 GT
        best_gt_idx = iou[best_query_idx].argmax()
        gt_box   = gt_boxes[best_gt_idx]           # [4] cxcywh
        pred_box = pred_boxes[b][best_query_idx]   # [4] cxcywh

        scores = pred_logits[b].max(dim=-1)[0]     # [N]

        # 1) CrossEntropy — Top-1 최적화
        total_cls += F.cross_entropy(scores.unsqueeze(0), best_query_idx.unsqueeze(0))

        # 2) L1 box loss
        total_l1 += F.l1_loss(pred_box, gt_box)

        # 3) GIoU loss
        pb_xyxy_sel = cxcywh_to_xyxy(pred_box.unsqueeze(0))
        gb_xyxy_sel = cxcywh_to_xyxy(gt_box.unsqueeze(0))
        giou = generalized_box_iou(pb_xyxy_sel, gb_xyxy_sel)
        total_giou += (1 - giou.diagonal()).mean()

        # 4) BCE — IoU > iou_pos_thr인 query 전부 positive → Top-5 최적화
        labels = (best_iou_per_query > iou_pos_thr).float()  # [N]
        n_pos = labels.sum().clamp(min=1)
        pos_weight = torch.tensor((labels.shape[0] - n_pos) / n_pos, device=scores.device)
        total_bce += F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)

        n_valid += 1

    n = max(n_valid, 1)
    return total_cls / n + w_l1 * total_l1 / n + w_giou * total_giou / n + w_bce * total_bce / n


def init_sa_from_pretrained(model, state_dict):
    """
    pretrained self_attn.in_proj_weight/bias → sa_q/k/v_proj
    pretrained self_attn.out_proj.*          → sa_out_proj
    """
    decoder_layers = model.transformer.decoder.layers
    d = decoder_layers[0].sa_q_proj.weight.shape[0]  # d_model

    for i, layer in enumerate(decoder_layers):
        prefix = f"transformer.decoder.layers.{i}.self_attn"

        w = state_dict.get(f"{prefix}.in_proj_weight")
        b = state_dict.get(f"{prefix}.in_proj_bias")
        ow = state_dict.get(f"{prefix}.out_proj.weight")
        ob = state_dict.get(f"{prefix}.out_proj.bias")

        if w is None:
            print(f"  [warn] self_attn weights not found for layer {i}, skipping")
            continue

        with torch.no_grad():
            layer.sa_q_proj.weight.copy_(w[:d, :])
            layer.sa_k_proj.weight.copy_(w[d:2*d, :])
            layer.sa_v_proj.weight.copy_(w[2*d:, :])
            if b is not None:
                layer.sa_q_proj.bias.copy_(b[:d])
                layer.sa_k_proj.bias.copy_(b[d:2*d])
                layer.sa_v_proj.bias.copy_(b[2*d:])
            layer.sa_out_proj.weight.copy_(ow)
            layer.sa_out_proj.bias.copy_(ob)

    print(f"  sa_q/k/v/out_proj initialized from pretrained self_attn ({len(decoder_layers)} layers)")


# ==========================================================
# Main
# ==========================================================
def main():
    device = "cuda"
    config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    pretrained_weights = "weights/groundingdino_swint_ogc.pth"

    args = SLConfig.fromfile(config_path)
    args.device = device

    model = build_groundingdino(args)
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = clean_state_dict(state_dict)  # remove "module." prefix if present

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights. Missing={len(missing)}, Unexpected={len(unexpected)}")

    # sa_q/k/v/out_proj를 pretrained self_attn 가중치로 초기화
    init_sa_from_pretrained(model, state_dict)

    model.to(device)

    # ==========================================================
    # backbone, text encoder, encoder frozen
    # decoder 전체 학습 (sa_rope는 lr 높게)
    # ==========================================================
    for param in model.parameters():
        param.requires_grad = False

    decoder_params = []
    rope_params = []
    for name, param in model.named_parameters():
        if "transformer.decoder" in name:
            param.requires_grad = True
            if "sa_rope" in name:
                rope_params.append(param)
            else:
                decoder_params.append(param)

    print(f"Decoder params (lr=1e-5): {sum(p.numel() for p in decoder_params):,}")
    print(f"RoPE params   (lr=5e-5): {sum(p.numel() for p in rope_params):,}")

    optimizer = optim.AdamW([
        {"params": decoder_params, "lr": 1e-5},
        {"params": rope_params,    "lr": 5e-5},
    ], weight_decay=1e-4)

    num_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)


    # ==========================================================
    # Data
    # ==========================================================
    dataset_path = "/data2/huggingface/AerialVG"
    image_dir    = "/data2/huggingface/AerialVG/images"

    ds = load_dataset(dataset_path)

    def _resize_keep_aspect(image, size=800, max_size=1333):
        w, h = image.size
        scale = size / min(w, h)
        if max(w, h) * scale > max_size:
            scale = max_size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return image.resize((new_w, new_h), PILImage.BILINEAR)

    transform = T.Compose([
        T.Lambda(lambda img: _resize_keep_aspect(img, size=800, max_size=1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = AerialVGDataset(ds["train"], image_dir=image_dir, transform=transform)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=4,
    )

    num_epochs = 15
    ckpt_dir = "weights/finetuned_rope_v8"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ==========================================================
    # Training loop
    # ==========================================================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        for batch_idx, (images, targets, texts) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images  = images.to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in t.items()} for t in targets]

            outputs = model(samples=images, targets=targets, captions=texts)

            pred_boxes   = outputs["pred_boxes"]    # [B, N, 4]
            pred_logits  = outputs["pred_logits"]   # [B, N, L]

            loss = ranking_loss(pred_boxes, pred_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder_params + rope_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if batch_idx % 100 == 0:
                pb0     = cxcywh_to_xyxy(pred_boxes[0])
                gb0     = cxcywh_to_xyxy(targets[0]["boxes"]) if targets[0]["boxes"].shape[0] > 0 else None
                max_iou = box_iou(pb0, gb0).max().item() if gb0 is not None else 0.0
                scores0 = pred_logits[0].max(dim=-1)[0]
                print(f"  [{batch_idx}/{len(train_loader)}] loss={loss.item():.4f}  "
                      f"max_iou={max_iou:.3f}  "
                      f"score=[{scores0.min().item():.2f}, {scores0.max().item():.2f}]")

        scheduler.step()
        avg_loss = running_loss / max(steps, 1)
        lr0 = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}  lr={lr0:.2e}")

        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:02d}.pth")
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"  Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
