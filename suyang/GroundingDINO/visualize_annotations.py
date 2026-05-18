"""
Visualize sample images with anchor + reference bboxes and relation labels.
Highlights consistent vs inconsistent cases (textual relation vs geometric angle).
"""
import json, math, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

VAL_ANNO = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT = "/data2/huggingface/AerialVG/images"
OUT_PATH = "/home/suyang0608/suyang/GroundingDINO/sample_annotations.png"

RELATION_ANGLE_RANGES = {
    "above":               (67.5, 112.5),
    "below":               (-112.5, -67.5),
    "on the left":         (157.5, 180, -180, -157.5),
    "on the right":        (-22.5, 22.5),
    "at the left":         (157.5, 180, -180, -157.5),
    "at the right":        (-22.5, 22.5),
    "at the top left":     (112.5, 157.5),
    "top left":            (112.5, 157.5),
    "at the top right":    (22.5, 67.5),
    "top right":           (22.5, 67.5),
    "at the bottom left":  (-157.5, -112.5),
    "bottom left":         (-157.5, -112.5),
    "at the bottom right": (-67.5, -22.5),
    "bottom right":        (-67.5, -22.5),
}

def normalize_relation(rel):
    rel = rel.lower().strip()
    if rel in RELATION_ANGLE_RANGES: return rel
    for k in RELATION_ANGLE_RANGES:
        if k in rel or rel in k: return k
    return None

def in_range(angle, ranges):
    if len(ranges) == 2: return ranges[0] <= angle <= ranges[1]
    if len(ranges) == 4: return (ranges[0] <= angle <= ranges[1]) or (ranges[2] <= angle <= ranges[3])
    return False

def bbox_center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)

def angle_deg(anchor_b, ref_b):
    ax, ay = bbox_center(anchor_b)
    rx, ry = bbox_center(ref_b)
    return math.degrees(math.atan2(-(ry-ay), rx-ax))

# ── Load and pick samples ─────────────────────────────────────────────────────
with open(VAL_ANNO) as f:
    metas = [json.loads(l) for l in f if l.strip()]

# Build sample lists by status
consistent_diag, inconsistent_diag, consistent_card = [], [], []
for m in metas:
    regions = m["grounding"]["regions"]
    if len(regions) < 2: continue
    anchor_b = regions[0]["bbox"]
    for ref in regions[1:]:
        if "realation" not in ref: continue
        rk = normalize_relation(ref["realation"])
        if rk is None: continue
        ang = angle_deg(anchor_b, ref["bbox"])
        ok  = in_range(ang, RELATION_ANGLE_RANGES[rk])
        is_diag = any(d in rk for d in ["top left","top right","bottom left","bottom right"])
        item = (m, ref, ang, rk)
        if is_diag and ok:        consistent_diag.append(item)
        elif is_diag and not ok:  inconsistent_diag.append(item)
        elif (not is_diag) and ok: consistent_card.append(item)

random.seed(2026)
picks = (
    [("consistent (cardinal)", x) for x in random.sample(consistent_card, 2)] +
    [("consistent (diagonal)", x) for x in random.sample(consistent_diag, 2)] +
    [("INCONSISTENT (diagonal)", x) for x in random.sample(inconsistent_diag, 2)]
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ax, (status, (m, ref, ang, rk)) in zip(axes, picks):
    img_path = f"{IMG_ROOT}/{m['filename']}"
    img = Image.open(img_path).convert("RGB")
    ax.imshow(img)

    anchor_b = m["grounding"]["regions"][0]["bbox"]
    anchor_phr = m["grounding"]["regions"][0].get("phrase", "anchor")
    ref_b      = ref["bbox"]
    ref_phr    = ref.get("phrase", "ref")
    rel_text   = ref["realation"]

    # anchor: blue thick
    ax.add_patch(patches.Rectangle(
        (anchor_b[0], anchor_b[1]), anchor_b[2]-anchor_b[0], anchor_b[3]-anchor_b[1],
        linewidth=3, edgecolor="#2196F3", facecolor="none"))
    ax.text(anchor_b[0], anchor_b[1]-5, f"ANCHOR: {anchor_phr}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#2196F3", alpha=0.9, pad=2, edgecolor="none"))

    # reference: red/green by consistency
    is_consistent = "INCONSISTENT" not in status
    ref_color = "#E53935" if not is_consistent else "#43A047"
    ax.add_patch(patches.Rectangle(
        (ref_b[0], ref_b[1]), ref_b[2]-ref_b[0], ref_b[3]-ref_b[1],
        linewidth=3, edgecolor=ref_color, facecolor="none", linestyle="--"))
    ax.text(ref_b[0], ref_b[1]-5, f"REF: {ref_phr}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=ref_color, alpha=0.9, pad=2, edgecolor="none"))

    # arrow from anchor center → ref center
    ac = bbox_center(anchor_b); rc = bbox_center(ref_b)
    ax.add_patch(FancyArrowPatch(ac, rc, arrowstyle="->", mutation_scale=20,
                                 color="yellow", linewidth=2.5))

    # title
    title  = f"[{status}]\n"
    title += f'label: "{rel_text}"   |   true angle: {ang:+.1f}°'
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=("#E53935" if not is_consistent else "#43A047"))
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Sample annotations — anchor (blue), reference (green=consistent, red=inconsistent)",
             fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
