import os
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import generalized_box_iou

from datasets import load_dataset

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from dpaa_adapter import insert_dpaa_adapters, freeze_except_dpaa, print_trainable_parameters
from feature_dpaa_adapter import insert_feature_dpaa, freeze_except_feature_dpaa, print_trainable_parameters as print_feature_trainable_parameters

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack(
        [
            cx - 0.5 * w,
            cy - 0.5 * h,
            cx + 0.5 * w,
            cy + 0.5 * h,
        ],
        dim=-1,
    )


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return torch.tensor(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ],
        dtype=torch.float32,
    )


def normalize_xyxy_box(box, width, height):
    box = torch.tensor(box, dtype=torch.float32)
    box[0::2] /= width
    box[1::2] /= height
    box = box.clamp(0, 1)
    return xyxy_to_cxcywh(box)


def find_image_path(local_path, filename):
    candidate_paths = [
        os.path.join(local_path, filename),
        os.path.join(local_path, "images", filename),
        os.path.join(local_path, "Images", filename),
        os.path.join(local_path, "train", filename),
        os.path.join(local_path, "validation", filename),
        os.path.join(local_path, "test", filename),
        os.path.join(local_path, "JPEGImages", filename),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Image file not found: {filename}\nChecked paths:\n" + "\n".join(candidate_paths)
    )


class AerialVGTrainDataset(Dataset):
    def __init__(
        self,
        local_path="/data2/huggingface/AerialVG",
        split="train",
        image_size=800,
        use_caption=False,
    ):
        self.local_path = local_path
        self.use_caption = use_caption

        ds = load_dataset(local_path)[split]
        self.samples = []

        for item in ds:
            filename = item["filename"]
            width = item["width"]
            height = item["height"]
            grounding = item["grounding"]

            caption = grounding.get("caption", "")
            regions = grounding.get("regions", [])

            for region in regions:
                if "bbox" not in region:
                    continue

                bbox = region["bbox"]
                phrase = region.get("phrase", "")

                if use_caption:
                    text = caption
                else:
                    text = phrase

                if not text:
                    continue

                self.samples.append(
                    {
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "text": text,
                        "bbox": bbox,
                    }
                )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = find_image_path(self.local_path, sample["filename"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        text = sample["text"].strip()
        if not text.endswith("."):
            text += "."

        target_box = normalize_xyxy_box(
            sample["bbox"],
            sample["width"],
            sample["height"],
        )

        return image_tensor, text, target_box


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    target_boxes = torch.stack([b[2] for b in batch], dim=0)
    return images, captions, target_boxes


def load_groundingdino(
    config_path,
    checkpoint_path,
    device,
    use_dpaa=False,
    dpaa_mid_dim=64,
    dpaa_kernel_l=15,
    dpaa_kernel_s=3,
    dpaa_scale=1.0,
    use_feature_dpaa=False,
    feature_dpaa_mid_dim=64,
    feature_dpaa_kernel_l=15,
    feature_dpaa_kernel_s=3,
    feature_dpaa_scale=1.0,
):
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device

    model = build_model(cfg)

    # 공식 GroundingDINO checkpoint는 먼저 plain model에 로드
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    print("[Plain GroundingDINO checkpoint load]")
    print(f"[Checkpoint] missing keys: {len(missing)}")
    print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

    # Backbone-DPAA 기존 실험용
    if use_dpaa:
        from dpaa_adapter import insert_dpaa_adapters, freeze_except_dpaa, print_trainable_parameters
        model = insert_dpaa_adapters(
            model,
            mid_dim=dpaa_mid_dim,
            kernel_l=dpaa_kernel_l,
            kernel_s=dpaa_kernel_s,
            scale=dpaa_scale,
            verbose=True,
        )
        model = freeze_except_dpaa(model)
        print_trainable_parameters(model)

    # FeatureEnhancer-DPAA 새 실험용
    if use_feature_dpaa:
        model = insert_feature_dpaa(
            model,
            mid_dim=feature_dpaa_mid_dim,
            kernel_l=feature_dpaa_kernel_l,
            kernel_s=feature_dpaa_kernel_s,
            scale=feature_dpaa_scale,
            verbose=True,
        )
        model = freeze_except_feature_dpaa(model)
        print_feature_trainable_parameters(model)

    model.to(device)

    return model



def compute_loss(outputs, target_boxes):
    """
    원래 안정화 버전.

    GT box와 가장 가까운 query를 bbox cost 기준으로 선택한 뒤,
    L1 + GIoU loss만 사용한다.
    """
    pred_boxes = outputs["pred_boxes"]  # [B, Q, 4], normalized cxcywh

    B, Q, _ = pred_boxes.shape
    selected_boxes = []

    for b in range(B):
        preds = pred_boxes[b]                  # [Q, 4]
        gt = target_boxes[b].unsqueeze(0)      # [1, 4]

        # L1 cost
        l1_cost = torch.cdist(preds, gt, p=1).squeeze(1)  # [Q]

        # GIoU cost
        pred_xyxy = box_cxcywh_to_xyxy(preds)
        gt_xyxy = box_cxcywh_to_xyxy(gt)
        giou = generalized_box_iou(pred_xyxy, gt_xyxy).squeeze(1)  # [Q]
        giou_cost = 1.0 - giou

        # bbox 기준으로 GT와 가장 가까운 query 선택
        cost = l1_cost + giou_cost
        best_idx = cost.argmin()

        selected_boxes.append(preds[best_idx])

    selected_boxes = torch.stack(selected_boxes, dim=0)

    loss_l1 = F.l1_loss(selected_boxes, target_boxes)

    pred_xyxy = box_cxcywh_to_xyxy(selected_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    giou = generalized_box_iou(pred_xyxy, target_xyxy)
    loss_giou = 1.0 - giou.diag().mean()

    loss = loss_l1 + loss_giou

    return loss, loss_l1.detach(), loss_giou.detach()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_path", default="/data2/huggingface/AerialVG")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--work_dir", default="work_dirs/aerialvg_swint")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--use_caption", action="store_true")
    parser.add_argument("--score_loss_weight", type=float, default=0.5)
    parser.add_argument("--rank_loss_weight", type=float, default=0.5)
    parser.add_argument("--rank_margin", type=float, default=0.2)

    
    parser.add_argument("--use_dpaa", action="store_true")
    parser.add_argument("--dpaa_mid_dim", type=int, default=64)
    parser.add_argument("--dpaa_kernel_l", type=int, default=15)
    parser.add_argument("--dpaa_kernel_s", type=int, default=3)
    parser.add_argument("--dpaa_scale", type=float, default=1.0)    

    parser.add_argument("--use_feature_dpaa", action="store_true")
    parser.add_argument("--feature_dpaa_mid_dim", type=int, default=64)
    parser.add_argument("--feature_dpaa_kernel_l", type=int, default=15)
    parser.add_argument("--feature_dpaa_kernel_s", type=int, default=3)
    parser.add_argument("--feature_dpaa_scale", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("Loading dataset...")
    dataset = AerialVGTrainDataset(
        local_path=args.local_path,
        split=args.split,
        image_size=args.image_size,
        use_caption=args.use_caption,
    )
    print("flattened train samples:", len(dataset))
    print("training prompt setting:", "caption" if args.use_caption else "phrase")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("Loading model...")
    model = load_groundingdino(
    config_path=args.config,
    checkpoint_path=args.checkpoint,
    device=device,
    use_dpaa=args.use_dpaa,
    dpaa_mid_dim=args.dpaa_mid_dim,
    dpaa_kernel_l=args.dpaa_kernel_l,
    dpaa_kernel_s=args.dpaa_kernel_s,
    dpaa_scale=args.dpaa_scale,
    use_feature_dpaa=args.use_feature_dpaa,
    feature_dpaa_mid_dim=args.feature_dpaa_mid_dim,
    feature_dpaa_kernel_l=args.feature_dpaa_kernel_l,
    feature_dpaa_kernel_s=args.feature_dpaa_kernel_s,
    feature_dpaa_scale=args.feature_dpaa_scale,
    )
    model.train()
    print("✅ 모델 로드 성공!")

    # ============================================================
    # Optimizer 설정
    # Detection Network 전체는 freeze.
    # use_feature_dpaa 실험에서는 오직 feature_dpaa 파라미터만 학습.
    # feature_dpaa.up 계층은 adapter 출력문 역할을 하므로 lr 10배 적용.
    # ============================================================
    if getattr(args, "use_feature_dpaa", False):
        up_params = []
        other_feature_dpaa_params = []

        print("\n[Optimizer parameter check: Feature-DPAA only]")
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            print(name)

            if "feature_dpaa" not in name:
                raise RuntimeError(f"Forbidden trainable parameter detected: {name}")

            if ".up." in name:
                up_params.append(p)
            else:
                other_feature_dpaa_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": other_feature_dpaa_params, "lr": args.lr},
                {"params": up_params, "lr": args.lr * 10.0},
            ],
            weight_decay=args.weight_decay,
        )

        print(f"[Optimizer] feature_dpaa other params lr = {args.lr}")
        print(f"[Optimizer] feature_dpaa up params lr = {args.lr * 10.0}")

    else:
        trainable_params = []

        print("\n[Optimizer parameter check]")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
                trainable_params.append(p)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        step_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, captions, target_boxes in pbar:
            images = images.to(device, non_blocking=True)
            target_boxes = target_boxes.to(device, non_blocking=True)

            outputs = model(images, captions=captions)
            loss, loss_l1, loss_giou = compute_loss(outputs, target_boxes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            global_step += 1
            step_count += 1

            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_giou += loss_giou.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "giou": f"{loss_giou.item():.4f}",
                    "step": global_step,
                    "lr": args.lr,
                }
            )

            if args.max_steps > 0 and step_count >= args.max_steps:
                break

        avg_loss = total_loss / max(1, step_count)
        avg_l1 = total_l1 / max(1, step_count)
        avg_giou = total_giou / max(1, step_count)

        print(
            f"[Epoch {epoch}] "
            f"avg_loss={avg_loss:.4f}, "
            f"avg_l1={avg_l1:.4f}, "
            f"avg_giou={avg_giou:.4f}"
        )

        if epoch % args.save_every == 0:
            save_path = os.path.join(args.work_dir, f"epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                save_path,
            )
            print("saved:", save_path)

        if args.max_steps > 0:
            print("max_steps mode finished.")
            break


if __name__ == "__main__":
    main()