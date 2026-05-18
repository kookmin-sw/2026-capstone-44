"""
Simple annotation visualization — just anchor + references with caption.
"""
import json, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

VAL_ANNO = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT = "/data2/huggingface/AerialVG/images"
OUT_PATH = "/home/suyang0608/suyang/GroundingDINO/sample_simple.png"

with open(VAL_ANNO) as f:
    metas = [json.loads(l) for l in f if l.strip()]

random.seed(42)
samples_with_refs = [m for m in metas
                     if any("realation" in r for r in m["grounding"]["regions"][1:])]
picks = random.sample(samples_with_refs, 6)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

ref_colors = ["#E53935", "#FB8C00", "#FDD835", "#7CB342"]

for ax, m in zip(axes, picks):
    img = Image.open(f"{IMG_ROOT}/{m['filename']}").convert("RGB")
    ax.imshow(img)

    regions = m["grounding"]["regions"]
    anchor = regions[0]
    ab = anchor["bbox"]

    # anchor: blue thick
    ax.add_patch(patches.Rectangle(
        (ab[0], ab[1]), ab[2]-ab[0], ab[3]-ab[1],
        linewidth=4, edgecolor="#2196F3", facecolor="none"))
    ax.text(ab[0], ab[1]-7, f"ANCHOR: {anchor.get('phrase','')}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="#2196F3", alpha=0.95, pad=3, edgecolor="none"))

    # references
    for i, ref in enumerate(regions[1:]):
        c = ref_colors[i % len(ref_colors)]
        rb = ref["bbox"]
        ax.add_patch(patches.Rectangle(
            (rb[0], rb[1]), rb[2]-rb[0], rb[3]-rb[1],
            linewidth=3, edgecolor=c, facecolor="none", linestyle="--"))
        rel = ref.get("realation", "—")
        label = f"REF: {ref.get('phrase','')}  ({rel})"
        ax.text(rb[0], rb[1]-7, label,
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor=c, alpha=0.95, pad=3, edgecolor="none"))

    ax.set_title(f'"{m["grounding"]["caption"]}"',
                 fontsize=10, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
