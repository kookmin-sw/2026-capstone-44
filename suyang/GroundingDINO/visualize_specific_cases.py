"""
Show 2 specific inconsistent cases:
  - "at the bottom right" labeled, true angle ~ -9.5° (almost due right)
  - "at the bottom right" labeled, true angle ~ -8.0° (almost due right)
With explanatory description on each panel.
"""
import json, math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

VAL_ANNO = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"
IMG_ROOT = "/data2/huggingface/AerialVG/images"
OUT_PATH = "/home/suyang0608/suyang/GroundingDINO/sample_specific_cases.png"

def bbox_center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def angle_deg(anchor_b, ref_b):
    ax, ay = bbox_center(anchor_b); rx, ry = bbox_center(ref_b)
    return math.degrees(math.atan2(-(ry-ay), rx-ax))

with open(VAL_ANNO) as f:
    metas = [json.loads(l) for l in f if l.strip()]

# Find "bottom right" cases with angle near -9.5 and -8.0
candidates = []
for m in metas:
    regions = m["grounding"]["regions"]
    if len(regions) < 2: continue
    anchor_b = regions[0]["bbox"]
    for ref in regions[1:]:
        if "realation" not in ref: continue
        rel = ref["realation"].lower()
        if "bottom right" not in rel: continue
        ang = angle_deg(anchor_b, ref["bbox"])
        # near 0 (almost due right): -15° < ang < 0°
        if -15 < ang < 0:
            candidates.append((m, ref, ang))

candidates.sort(key=lambda x: -x[2])  # closest to 0° first
print(f"Found {len(candidates)} candidates with bottom-right label and angle in (-15, 0)")
for c in candidates[:5]:
    print(f"  angle={c[2]:.1f}° file={c[0]['filename']}")

# Pick top 2 (closest to 0°, i.e. most clearly "almost due right")
picks = candidates[:2]

fig, axes = plt.subplots(1, 2, figsize=(24, 11))
for ax, (m, ref, ang) in zip(axes, picks):
    img = Image.open(f"{IMG_ROOT}/{m['filename']}").convert("RGB")
    ax.imshow(img)

    anchor_b   = m["grounding"]["regions"][0]["bbox"]
    anchor_phr = m["grounding"]["regions"][0].get("phrase", "anchor")
    ref_b      = ref["bbox"]
    ref_phr    = ref.get("phrase", "ref")

    # anchor: blue
    ax.add_patch(patches.Rectangle(
        (anchor_b[0], anchor_b[1]), anchor_b[2]-anchor_b[0], anchor_b[3]-anchor_b[1],
        linewidth=5, edgecolor="#2196F3", facecolor="none"))
    ax.text(anchor_b[0], anchor_b[1]-12, f"ANCHOR: {anchor_phr}",
            color="white", fontsize=18, fontweight="bold",
            bbox=dict(facecolor="#2196F3", alpha=0.95, pad=5, edgecolor="none"))

    # reference: red
    ax.add_patch(patches.Rectangle(
        (ref_b[0], ref_b[1]), ref_b[2]-ref_b[0], ref_b[3]-ref_b[1],
        linewidth=5, edgecolor="#E53935", facecolor="none", linestyle="--"))
    ax.text(ref_b[0], ref_b[1]-12, f"REF: {ref_phr}",
            color="white", fontsize=18, fontweight="bold",
            bbox=dict(facecolor="#E53935", alpha=0.95, pad=5, edgecolor="none"))

    # arrow
    ac = bbox_center(anchor_b); rc = bbox_center(ref_b)
    ax.add_patch(FancyArrowPatch(ac, rc, arrowstyle="->", mutation_scale=35,
                                 color="yellow", linewidth=5))

    rel_text = ref["realation"]
    title  = "INCONSISTENT (diagonal)\n"
    title += f'label: "{rel_text}"   |   true angle: {ang:+.1f}°'
    ax.set_title(title, fontsize=22, fontweight="bold", color="#E53935")
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Inconsistent diagonal cases — textual label vs. true geometric angle",
             fontsize=20, fontweight="bold", y=0.99)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
print(f"\nSaved: {OUT_PATH}")
