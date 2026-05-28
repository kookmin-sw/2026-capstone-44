import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from cached_dataset import CachedTopKDataset, cached_collate_fn
from reranker import AerialReranker

BLUE   = "#2563EB"
ORANGE = "#F97316"
GREEN  = "#22C55E"
BG     = "#FFFFFF"
CARD   = "#FFFFFF"
TEXT   = "#000000"
MUTED  = "#444444"

def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

def compute_iou(box1, box2):
    b1 = cxcywh_to_xyxy(box1)
    b2 = cxcywh_to_xyxy(box2)
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (b1[..., 2] - b1[..., 0]).clamp(0) * (b1[..., 3] - b1[..., 1]).clamp(0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(0) * (b2[..., 3] - b2[..., 1]).clamp(0)
    return (inter / (area1 + area2 - inter + 1e-6)).item()

def draw_bbox(ax, box_cxcywh, img_w, img_h, color, label, linewidth=2.5, linestyle="-"):
    cx, cy, bw, bh = box_cxcywh.tolist()
    x1 = (cx - bw/2) * img_w
    y1 = (cy - bh/2) * img_h
    w  = bw * img_w
    h  = bh * img_h
    rect = patches.FancyBboxPatch(
        (x1, y1), w, h,
        boxstyle="round,pad=1",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        zorder=5,
    )
    ax.add_patch(rect)
    if label == "GT":
        tx, ha = x1 + w - 4, "right"
    else:
        tx, ha = x1 + 4, "left"
    ax.text(
        tx, y1 + 14, label,
        color="white", fontsize=7, fontweight="bold",
        ha=ha,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, linewidth=0),
        zorder=6,
    )

def save_performance_bar(baseline_top1, baseline_top5, ours_top1, ours_top5, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    metrics = [("Top-1 Accuracy", baseline_top1, ours_top1),
               ("Top-5 Accuracy", baseline_top5, ours_top5)]
    for ax, (title, base_val, our_val) in zip(axes, metrics):
        ax.set_facecolor(CARD)
        bars = ax.bar(
            ["GroundingDINO\n(Baseline)", "KRG\n(Ours)"],
            [base_val * 100, our_val * 100],
            color=[BLUE, ORANGE], width=0.45, zorder=3, edgecolor="none",
        )
        for bar, val in zip(bars, [base_val, our_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val*100:.2f}%",
                ha="center", va="bottom",
                color=TEXT, fontsize=12, fontweight="bold",
            )
        improve = (our_val - base_val) * 100
        sign = "+" if improve >= 0 else ""
        ax.annotate(
            f"{sign}{improve:.2f}%",
            xy=(1, our_val * 100),
            xytext=(0.5, max(base_val, our_val) * 100 + 3),
            ha="center", color=GREEN, fontsize=11, fontweight="bold",
        )
        ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Accuracy (%)", color=MUTED, fontsize=10)
        ax.tick_params(colors=MUTED)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.yaxis.grid(True, color="#334155", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ymax = max(base_val, our_val) * 100
        ax.set_ylim(0, ymax * 1.25)
        for label in ax.get_xticklabels():
            label.set_color(TEXT)
            label.set_fontsize(10)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE,   label="GroundingDINO (Baseline)"),
        Patch(facecolor=ORANGE, label="KRG (Ours)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               frameon=False, labelcolor=TEXT, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "AerialVG Zero-Shot Evaluation  ·  KRG vs GroundingDINO",
        color=TEXT, fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {save_path}")

def save_qualitative_grid(samples, save_path):
    N = len(samples)
    fig = plt.figure(figsize=(14, 5 * N), facecolor=BG)
    fig.suptitle(
        "Qualitative Comparison  ·  GroundingDINO vs KRG",
        color=TEXT, fontsize=15, fontweight="bold", y=1.005,
    )
    outer = gridspec.GridSpec(N, 2, figure=fig, hspace=0.35, wspace=0.08)
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        img_w, img_h = image.size
        img_arr = np.array(image)
        for col, (method, box, iou, color, label) in enumerate([
            ("GroundingDINO (Baseline)", sample["baseline_box"], sample["baseline_iou"], BLUE,   "Baseline"),
            ("KRG (Ours)",              sample["ours_box"],      sample["ours_iou"],     ORANGE, "KRG"),
        ]):
            ax = fig.add_subplot(outer[i, col])
            ax.imshow(img_arr)
            ax.set_facecolor(BG)
            draw_bbox(ax, sample["gt_box"], img_w, img_h, GREEN, "GT", 2.0)
            draw_bbox(ax, box, img_w, img_h, color, label, 2.5)
            iou_color = GREEN if iou >= 0.5 else "#EF4444"
            ax.text(
                0.98, 0.04, f"IoU: {iou:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=1.5", facecolor=iou_color, alpha=0.9, linewidth=0),
                zorder=7,
            )
            ax.set_title(method, color=TEXT, fontsize=10, fontweight="bold", pad=6)
            ax.axis("off")
            if col == 0:
                improve_sign = "▲" if sample["improvement"] > 0 else "▼"
                improve_color = GREEN if sample["improvement"] > 0 else "#EF4444"
                ax.text(
                    -0.04, 0.5,
                    f"#{i+1}\n{improve_sign}{abs(sample['improvement']):.3f}",
                    transform=ax.transAxes, ha="right", va="center",
                    color=improve_color, fontsize=8, fontweight="bold",
                )
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {save_path}")

def save_single_comparison(sample, save_path, title=""):
    image = Image.open(sample["image_path"]).convert("RGB")
    img_w, img_h = image.size
    img_arr = np.array(image)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=BG)
    caption = sample.get("caption", "")
    full_title = f"{title}\n\"{caption}\"" if caption else title
    fig.suptitle(full_title, color=TEXT, fontsize=11, fontweight="bold")
    for ax, (method, box, iou, color, label) in zip(axes, [
        ("GroundingDINO (Baseline)", sample["baseline_box"], sample["baseline_iou"], BLUE,   "Baseline"),
        ("KRG (Ours)",              sample["ours_box"],      sample["ours_iou"],     ORANGE, "KRG"),
    ]):
        ax.imshow(img_arr)
        ax.set_facecolor(BG)
        draw_bbox(ax, sample["gt_box"], img_w, img_h, GREEN, "GT", 1.5)
        draw_bbox(ax, box, img_w, img_h, color, label, 2.5)
        iou_color = GREEN if iou >= 0.5 else "#EF4444"
        ax.text(
            0.98, 0.04, f"IoU: {iou:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color="white", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=1.5", facecolor=iou_color, alpha=0.9, linewidth=0),
            zorder=7,
        )
        ax.set_title(method, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[saved] {save_path}")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_cache",   required=True)
    parser.add_argument("--ckpt_path",   required=True)
    parser.add_argument("--image_root",  required=True)
    parser.add_argument("--save_dir",    default="outputs/visualization")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--lambda_r",    type=float, default=1.0)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt      = torch.load(args.ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    knn_k     = saved_args.get("knn_k", 5)

    reranker = AerialReranker(
        logit_dim=256, hidden_dim=256, num_layers=2, nhead=4, knn_k=knn_k,
    ).to(device)
    reranker.load_state_dict(ckpt["reranker"] if "reranker" in ckpt else ckpt)
    reranker.eval()

    dataset = CachedTopKDataset(args.val_cache)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=cached_collate_fn)

    BASELINE_TOP1 = 0.1276730891382596
    BASELINE_TOP5 = 0.34490789752276096

    total = top1_correct = top5_correct = 0
    all_samples = []

    for batch in loader:
        topk_logits = batch["topk_logits"].to(device).float()
        topk_boxes  = batch["topk_boxes"].to(device).float()
        topk_scores = batch["topk_scores"].to(device).float()
        gt_boxes    = batch["gt_boxes"].to(device).float()

        rerank_raw   = reranker(topk_logits, topk_boxes, topk_scores)
        rerank_score = torch.sigmoid(rerank_raw)
        final_scores = topk_scores + args.lambda_r * rerank_score
        sorted_idx   = torch.argsort(final_scores, dim=1, descending=True).cpu()
        baseline_idx = torch.argmax(topk_scores, dim=1).cpu()
        B = topk_boxes.size(0)

        for b in range(B):
            gt    = gt_boxes[b].cpu()
            boxes = topk_boxes[b].cpu()
            top1_box = boxes[sorted_idx[b, 0]].cpu()
            base_box = boxes[baseline_idx[b]].cpu()
            ours_iou = compute_iou(top1_box, gt)
            base_iou = compute_iou(base_box, gt)
            top5_boxes = boxes[sorted_idx[b, :5]]
            top5_iou   = max(compute_iou(top5_boxes[j], gt) for j in range(len(top5_boxes)))
            top1_correct += int(ours_iou >= 0.5)
            top5_correct += int(top5_iou >= 0.5)
            total        += 1

            if "image_path" in batch:
                img_path = batch["image_path"][b]
                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(os.path.join(CURRENT_DIR, img_path))
                if os.path.exists(img_path):
                    all_samples.append({
                        "image_path":   img_path,
                        "gt_box":       gt,
                        "baseline_box": base_box,
                        "ours_box":     top1_box,
                        "baseline_iou": base_iou,
                        "ours_iou":     ours_iou,
                        "improvement":  ours_iou - base_iou,
                        "caption":      batch.get("caption", [""])[b] if "caption" in batch else "",
                    })

    ours_top1 = top1_correct / total
    ours_top5 = top5_correct / total

    print("\n" + "="*55)
    print("   AerialVG Zero-Shot Evaluation  ·  KRG")
    print("="*55)
    print(f"\n[Baseline: GroundingDINO]")
    print(f"  Top-1 : {BASELINE_TOP1*100:.2f}%")
    print(f"  Top-5 : {BASELINE_TOP5*100:.2f}%")
    print(f"\n[Ours: KRG (K-Nearest Relation Grounding)]")
    print(f"  Top-1 : {ours_top1*100:.2f}%")
    print(f"  Top-5 : {ours_top5*100:.2f}%")
    print(f"\n[Improvement]")
    print(f"  Top-1 : +{(ours_top1 - BASELINE_TOP1)*100:.2f}%")
    print(f"  Top-5 : +{(ours_top5 - BASELINE_TOP5)*100:.2f}%")
    print("="*55)

    save_performance_bar(
        BASELINE_TOP1, BASELINE_TOP5, ours_top1, ours_top5,
        os.path.join(args.save_dir, "performance_bar.png"),
    )

    if len(all_samples) == 0:
        print("\n[Warning] image_path가 cache에 없어서 qualitative 시각화 생략.")
        return

    improved = sorted(all_samples, key=lambda x: x["improvement"], reverse=True)
    failed   = sorted(all_samples, key=lambda x: x["improvement"])
    top_improved = [s for s in improved if s["improvement"] > 0][:args.num_samples]
    top_failed   = [s for s in failed   if s["improvement"] < 0][:3]

    if top_improved:
        save_qualitative_grid(top_improved, os.path.join(args.save_dir, "qualitative_improved.png"))
    if top_failed:
        save_qualitative_grid(top_failed,   os.path.join(args.save_dir, "qualitative_failed.png"))

    for rank, sample in enumerate(top_improved[:3]):
        save_single_comparison(
            sample,
            os.path.join(args.save_dir, f"success_{rank+1}.png"),
            title=f"Success Case #{rank+1}  |  IoU: {sample['baseline_iou']:.3f} → {sample['ours_iou']:.3f}  (+{sample['improvement']:.3f})",
        )
    for rank, sample in enumerate(top_failed[:2]):
        save_single_comparison(
            sample,
            os.path.join(args.save_dir, f"failure_{rank+1}.png"),
            title=f"Failure Case #{rank+1}  |  IoU: {sample['baseline_iou']:.3f} → {sample['ours_iou']:.3f}  ({sample['improvement']:.3f})",
        )

    print(f"\n모든 시각화 저장 완료: {args.save_dir}")

if __name__ == "__main__":
    main()
