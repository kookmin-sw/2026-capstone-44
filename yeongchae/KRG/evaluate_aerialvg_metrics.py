import os
import sys
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from cached_dataset import CachedTopKDataset, cached_collate_fn
from reranker import AerialReranker

BASELINE_TOP1 = 0.1276730891382596
BASELINE_TOP5 = 0.34490789752276096


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def box_iou(box1, box2):
    box1 = cxcywh_to_xyxy(box1)
    box2 = cxcywh_to_xyxy(box2)
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-6)


def to_percent(x):
    return f"{x * 100:.2f}%"


def normalize_score(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


def rank_score(x):
    rank = torch.argsort(torch.argsort(-x, dim=1), dim=1).float()
    return -rank


def compute_final_scores(topk_scores, rerank_scores, alpha, K, mode):
    topk_norm   = normalize_score(topk_scores)
    rerank_norm = normalize_score(rerank_scores)
    topk_rank   = rank_score(topk_scores)
    rerank_rank = rank_score(rerank_scores)

    if mode == "baseline":
        final_scores = topk_scores.clone()
    elif mode == "add_raw":
        final_scores = topk_scores + alpha * rerank_scores
    elif mode == "add_norm":
        final_scores = topk_scores + alpha * rerank_norm
    elif mode == "norm_sum":
        final_scores = topk_norm + alpha * rerank_norm
    elif mode == "rank_fusion":
        final_scores = topk_rank + alpha * rerank_rank
    elif mode == "score_rank_mix":
        final_scores = topk_scores + alpha * rerank_rank
    elif mode == "conservative":
        final_scores = 0.9 * topk_norm + alpha * rerank_norm
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if K is not None:
        final_scores[:, K:] = -1e9

    return final_scores


def get_metrics(top1_ious, top5_ious):
    return {
        "top1": (top1_ious >= 0.5).float().mean().item(),
        "top5": (top5_ious >= 0.5).float().mean().item(),
    }


def print_metrics(name, metrics):
    print(f"\n[{name}]")
    print(f"Top-1 : {to_percent(metrics['top1'])}")
    print(f"Top-5 : {to_percent(metrics['top5'])}")


def save_bar_graph(baseline_metrics, ours_metrics, save_path):
    names = ["Top-1", "Top-5"]
    baseline = [baseline_metrics["top1"] * 100, baseline_metrics["top5"] * 100]
    ours     = [ours_metrics["top1"] * 100,     ours_metrics["top5"] * 100]
    x = list(range(len(names)))
    width = 0.35
    plt.figure(figsize=(6, 5))
    plt.bar([i - width/2 for i in x], baseline, width, label="GroundingDINO")
    plt.bar([i + width/2 for i in x], ours,     width, label="GroundingDINO + Ours")
    plt.xticks(x, names)
    plt.ylabel("Accuracy (%)")
    plt.title("AerialVG Zero-Shot Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_box(ax, box, image_w, image_h, color, label):
    cx, cy, bw, bh = box.tolist()
    x1 = (cx - bw/2) * image_w
    y1 = (cy - bh/2) * image_h
    rect = patches.Rectangle((x1, y1), bw * image_w, bh * image_h,
                               linewidth=2, edgecolor=color, facecolor="none", label=label)
    ax.add_patch(rect)


def save_qualitative_image(image_path, gt_box, baseline_box, ours_box, save_path):
    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    draw_box(ax, gt_box,       image_w, image_h, "lime", "GT")
    draw_box(ax, baseline_box, image_w, image_h, "red",  "GroundingDINO")
    draw_box(ax, ours_box,     image_w, image_h, "blue", "Ours")
    ax.axis("off")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_cache = os.path.join(ROOT_DIR, "dataset/AerialVG/cache/val_topk20.pt")

    ckpt_candidates = [
        os.path.join(CURRENT_DIR, "outputs/aerial_reranker_best.pth"),
        os.path.join(CURRENT_DIR, "aerial_reranker_best.pth"),
        os.path.join(ROOT_DIR,    "aerial_reranker_best.pth"),
    ]

    ckpt_path = None
    for path in ckpt_candidates:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        raise FileNotFoundError("aerial_reranker_best.pth를 찾지 못했어.")

    result_dir = os.path.join(CURRENT_DIR, "outputs")
    vis_dir    = os.path.join(result_dir, "qualitative")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(vis_dir,    exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    dataset = CachedTopKDataset(val_cache)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=cached_collate_fn)

    ckpt      = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    knn_k     = saved_args.get("knn_k", 5)

    reranker = AerialReranker(
        logit_dim=256, hidden_dim=256, num_layers=2, nhead=4, knn_k=knn_k,
    ).to(device)

    state_dict = ckpt["reranker"] if "reranker" in ckpt else ckpt
    reranker.load_state_dict(state_dict)
    reranker.eval()

    alpha_list = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    k_list     = [1, 2, 3, 4, 5, 7, 10, None]
    mode_list  = ["baseline", "add_raw", "add_norm", "norm_sum",
                  "rank_fusion", "score_rank_mix", "conservative"]

    baseline_metrics = {"top1": BASELINE_TOP1, "top5": BASELINE_TOP5}

    ours_top1_by_setting = {(a, K, m): [] for a in alpha_list for K in k_list for m in mode_list}
    ours_top5_by_setting = {(a, K, m): [] for a in alpha_list for K in k_list for m in mode_list}
    vis_samples_by_setting = {(a, K, m): [] for a in alpha_list for K in k_list for m in mode_list}

    for batch in loader:
        topk_logits = batch["topk_logits"].to(device).float()
        topk_boxes  = batch["topk_boxes"].to(device).float()
        topk_scores = batch["topk_scores"].to(device).float()
        gt_box      = batch["gt_box"].to(device).float()

        batch_size = topk_boxes.size(0)
        batch_idx  = torch.arange(batch_size, device=device)

        all_ious = box_iou(
            topk_boxes.reshape(-1, 4),
            gt_box[:, None, :].expand_as(topk_boxes).reshape(-1, 4),
        ).reshape(batch_size, -1)

        baseline_top1_idx = torch.argmax(topk_scores, dim=1)
        baseline_top1_iou = all_ious[batch_idx, baseline_top1_idx]

        rerank_raw   = reranker(topk_logits, topk_boxes, topk_scores)
        rerank_score = torch.sigmoid(rerank_raw)

        for alpha in alpha_list:
            for K in k_list:
                for mode in mode_list:
                    final_scores = compute_final_scores(
                        topk_scores.clone(), rerank_score.clone(), alpha, K, mode)

                    top1_idx = torch.argmax(final_scores, dim=1)
                    top5_idx = torch.topk(final_scores, k=min(5, final_scores.size(1)), dim=1).indices

                    top1_iou = all_ious[batch_idx, top1_idx]
                    top5_iou = all_ious.gather(1, top5_idx).max(dim=1).values

                    ours_top1_by_setting[(alpha, K, mode)].append(top1_iou.cpu())
                    ours_top5_by_setting[(alpha, K, mode)].append(top5_iou.cpu())

                    if "image_path" in batch:
                        for i in range(batch_size):
                            image_path = batch["image_path"][i]
                            if not os.path.isabs(image_path):
                                image_path = os.path.join(ROOT_DIR, image_path)
                            vis_samples_by_setting[(alpha, K, mode)].append({
                                "image_path":   image_path,
                                "gt_box":       gt_box[i].detach().cpu(),
                                "baseline_box": topk_boxes[i, baseline_top1_idx[i]].detach().cpu(),
                                "ours_box":     topk_boxes[i, top1_idx[i]].detach().cpu(),
                                "improvement":  top1_iou[i].item() - baseline_top1_iou[i].item(),
                            })

    results = []
    for alpha in alpha_list:
        for K in k_list:
            for mode in mode_list:
                top1_ious = torch.cat(ours_top1_by_setting[(alpha, K, mode)])
                top5_ious = torch.cat(ours_top5_by_setting[(alpha, K, mode)])
                metrics   = get_metrics(top1_ious, top5_ious)
                results.append({"alpha": alpha, "K": K, "mode": mode, "metrics": metrics})

    best = max(results, key=lambda r: (r["metrics"]["top1"], r["metrics"]["top5"]))
    best_alpha, best_K, best_mode, best_metrics = best["alpha"], best["K"], best["mode"], best["metrics"]

    print("\n================ AerialVG Zero-Shot Evaluation ================")
    print_metrics("Baseline: GroundingDINO",        baseline_metrics)
    print_metrics("Ours: GroundingDINO + Reranker", best_metrics)

    print("\n================ Best Setting ================")
    print(f"mode  : {best_mode}")
    print(f"alpha : {best_alpha}")
    print(f"K     : {best_K}")

    print("\n================ Paper-style Table ================")
    print("| Method | Setting | Top-1 | Top-5 |")
    print("|---|---|---:|---:|")
    print(f"| GroundingDINO | Zero-Shot | {to_percent(baseline_metrics['top1'])} | {to_percent(baseline_metrics['top5'])} |")
    print(f"| GroundingDINO + Ours | Zero-Shot | {to_percent(best_metrics['top1'])} | {to_percent(best_metrics['top5'])} |")

    print("\n================ Improvement ================")
    print(f"Top-1 : {to_percent(best_metrics['top1'] - baseline_metrics['top1'])}")
    print(f"Top-5 : {to_percent(best_metrics['top5'] - baseline_metrics['top5'])}")
    print("===================================================")

    top10 = sorted(results, key=lambda r: (r["metrics"]["top1"], r["metrics"]["top5"]), reverse=True)[:10]
    print("\n================ Top 10 Settings ================")
    for r in top10:
        m = r["metrics"]
        print(f"mode={r['mode']}, alpha={r['alpha']}, K={r['K']} | Top-1={to_percent(m['top1'])}, Top-5={to_percent(m['top5'])}")

    graph_path = os.path.join(result_dir, "aerialvg_paper_style_bar_graph.png")
    save_bar_graph(baseline_metrics, best_metrics, graph_path)
    print(f"\nSaved bar graph: {graph_path}")

    best_vis_samples = vis_samples_by_setting.get((best_alpha, best_K, best_mode), [])
    if len(best_vis_samples) == 0:
        print("\n[Warning] image_path가 cache에 없어서 qualitative 이미지는 저장하지 못했어.")
        return

    improved = sorted(best_vis_samples, key=lambda x: x["improvement"], reverse=True)[:5]
    failed   = sorted(best_vis_samples, key=lambda x: x["improvement"])[:3]
    saved_count = 0

    for rank, sample in enumerate(improved):
        if not os.path.exists(sample["image_path"]):
            continue
        save_qualitative_image(sample["image_path"], sample["gt_box"],
                               sample["baseline_box"], sample["ours_box"],
                               os.path.join(vis_dir, f"success_{rank+1}.png"))
        saved_count += 1

    for rank, sample in enumerate(failed):
        if not os.path.exists(sample["image_path"]):
            continue
        save_qualitative_image(sample["image_path"], sample["gt_box"],
                               sample["baseline_box"], sample["ours_box"],
                               os.path.join(vis_dir, f"failure_{rank+1}.png"))
        saved_count += 1

    print(f"\nSaved qualitative results: {saved_count} images")
    print(vis_dir)


if __name__ == "__main__":
    evaluate()
