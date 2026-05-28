import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from cached_dataset import CachedTopKDataset, cached_collate_fn
from reranker import AerialReranker


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        B, K = inputs.shape
        target_onehot = torch.zeros_like(inputs)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        prob = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction="none")

        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        loss = alpha_t * focal_weight * bce

        return loss.mean()


def train_one_epoch(reranker, loader, optimizer, criterion, device):
    reranker.train()
    total_loss = 0.0
    total_acc = 0
    total_count = 0

    for batch in tqdm(loader, desc="train"):
        topk_logits = batch["topk_logits"].to(device)
        topk_boxes  = batch["topk_boxes"].to(device)
        topk_scores = batch["topk_scores"].to(device)
        labels      = batch["labels"].to(device)

        rerank_scores = reranker(topk_logits, topk_boxes, topk_scores)
        loss = criterion(rerank_scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = rerank_scores.argmax(dim=1)
        total_loss  += loss.item() * labels.size(0)
        total_acc   += (pred == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_acc / total_count


@torch.no_grad()
def evaluate(reranker, loader, device, lambda_r=1.0):
    reranker.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    mean_iou = 0.0

    for batch in tqdm(loader, desc="eval"):
        topk_logits = batch["topk_logits"].to(device)
        topk_boxes  = batch["topk_boxes"].to(device)
        topk_scores = batch["topk_scores"].to(device)
        gt_boxes    = batch["gt_boxes"].to(device)

        rerank_raw   = reranker(topk_logits, topk_boxes, topk_scores)
        rerank_score = torch.sigmoid(rerank_raw)

        final_scores = topk_scores + lambda_r * rerank_score
        sorted_idx   = torch.argsort(final_scores, dim=1, descending=True)

        B = topk_boxes.size(0)

        for b in range(B):
            pred_xyxy = box_convert(topk_boxes[b], "cxcywh", "xyxy")
            gt_xyxy   = box_convert(gt_boxes[b].unsqueeze(0), "cxcywh", "xyxy")
            ious = box_iou(pred_xyxy, gt_xyxy).squeeze(1)

            top1_idx = sorted_idx[b, 0]
            top1_iou = ious[top1_idx].item()

            top5_idx = sorted_idx[b, :5]
            top5_iou = ious[top5_idx].max().item()

            top1_correct += int(top1_iou >= 0.5)
            top5_correct += int(top5_iou >= 0.5)
            mean_iou     += top1_iou
            total        += 1

    return {
        "Top1":    top1_correct / total,
        "Top5":    top5_correct / total,
        "MeanIoU": mean_iou / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache",   required=True)
    parser.add_argument("--topk",        type=int,   default=20)
    parser.add_argument("--knn_k",       type=int,   default=5)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lambda_r",    type=float, default=1.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--save_path",   default="aerial_reranker_best.pth")
    args = parser.parse_args()

    train_set = CachedTopKDataset(args.train_cache)
    val_set   = CachedTopKDataset(args.val_cache)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=cached_collate_fn, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=cached_collate_fn, num_workers=2)

    reranker = AerialReranker(
        logit_dim=256, hidden_dim=256, num_layers=2, nhead=4, knn_k=args.knn_k,
    ).to(args.device)

    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    best_top1 = 0.0

    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(reranker, train_loader, optimizer, criterion, args.device)
        metrics   = evaluate(reranker, val_loader, args.device, args.lambda_r)

        print(
            f"[Epoch {epoch + 1}] "
            f"loss={loss:.4f} "
            f"train_acc={acc:.4f} "
            f"Top1={metrics['Top1']:.4f} "
            f"Top5={metrics['Top5']:.4f} "
            f"MeanIoU={metrics['MeanIoU']:.4f}"
        )

        if metrics["Top1"] > best_top1:
            best_top1 = metrics["Top1"]
            torch.save(
                {"reranker": reranker.state_dict(), "epoch": epoch + 1,
                 "best_top1": best_top1, "metrics": metrics, "args": vars(args)},
                args.save_path,
            )
            print(f"Best model saved: {args.save_path}")


if __name__ == "__main__":
    main()
