"""
Analyze geometric angle distribution of (anchor, reference) pairs vs textual relation.
Outputs histogram of inconsistency by angle range.
"""
import json, math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

VAL_ANNO = "/data2/huggingface/AerialVG/annotation/vg_val_odvg.jsonl"

# Map textual relation → expected angle range (math convention, atan2(-dy, dx))
# Strict 8-way bucketing: each direction = 45° wide, non-overlapping, full 360° coverage
# (vs. earlier 60° loose windows → 17.5%; this strict version → ~28%)
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

def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def angle_deg(anchor_bbox, ref_bbox):
    ax, ay = bbox_center(anchor_bbox)
    rx, ry = bbox_center(ref_bbox)
    dx, dy = rx - ax, ry - ay
    # image y is down → flip for math convention so "above" = +90°
    return math.degrees(math.atan2(-dy, dx))

def in_range(angle, ranges):
    if len(ranges) == 2:
        return ranges[0] <= angle <= ranges[1]
    if len(ranges) == 4:
        return (ranges[0] <= angle <= ranges[1]) or (ranges[2] <= angle <= ranges[3])
    return False

def normalize_relation(rel):
    rel = rel.lower().strip()
    if rel in RELATION_ANGLE_RANGES:
        return rel
    for k in RELATION_ANGLE_RANGES:
        if k in rel or rel in k:
            return k
    return None


# ── Load + analyze ─────────────────────────────────────────────────────────────
with open(VAL_ANNO) as f:
    metas = [json.loads(l) for l in f if l.strip()]

all_angles, consistent_flags, relation_keys = [], [], []
unmapped = defaultdict(int)

for m in metas:
    regions = m["grounding"]["regions"]
    if not regions: continue
    anchor_bbox = regions[0]["bbox"]
    for ref in regions[1:]:
        if "realation" not in ref: continue
        rel_text = ref["realation"]
        rel_key  = normalize_relation(rel_text)
        if rel_key is None:
            unmapped[rel_text] += 1
            continue
        ang = angle_deg(anchor_bbox, ref["bbox"])
        all_angles.append(ang)
        consistent_flags.append(in_range(ang, RELATION_ANGLE_RANGES[rel_key]))
        relation_keys.append(rel_key)

all_angles = np.array(all_angles)
consistent = np.array(consistent_flags)
total = len(all_angles)
inconsistent_rate = (~consistent).sum() / total

print(f"Total (anchor,ref) pairs with mappable relation: {total}")
print(f"Consistent  : {consistent.sum()}  ({100*consistent.mean():.1f}%)")
print(f"Inconsistent: {(~consistent).sum()}  ({100*inconsistent_rate:.1f}%)")
if unmapped:
    print("Unmapped relation strings (not analyzed):")
    for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:10]:
        print(f"  '{k}': {v}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
fig.suptitle("Label Accuracy Analysis — AerialVG textual relation vs. true geometric angle",
             fontsize=14, fontweight="bold", y=1.02)

# (1) Overall histogram
ax = axes[0]
bins = np.arange(-180, 181, 10)
ax.hist(all_angles[consistent],  bins=bins, color="#4CAF50", alpha=0.85,
        label=f"consistent ({consistent.sum()})")
ax.hist(all_angles[~consistent], bins=bins, color="#E53935", alpha=0.85,
        label=f"inconsistent ({(~consistent).sum()})")
ax.set_xlabel("Geometric angle (deg, +x→0°, +y(up)→90°)")
ax.set_ylabel("Count")
ax.set_title(f"Angle distribution — Inconsistency: {100*inconsistent_rate:.1f}% (strict 45° per 8-way bucket)")
ax.set_xticks(np.arange(-180, 181, 45))
ax.legend()
ax.grid(alpha=0.3)
# guide lines for cardinal directions
for deg, lbl in [(90,"above"),(0,"right"),(-90,"below"),(180,"left"),(-180,"left")]:
    ax.axvline(deg, color="gray", lw=0.5, ls="--")

# (2) Inconsistency rate by relation type
ax = axes[1]
rel_stats = defaultdict(lambda: [0, 0])  # [total, inconsistent]
for k, c in zip(relation_keys, consistent):
    rel_stats[k][0] += 1
    if not c: rel_stats[k][1] += 1

# canonicalize duplicate "at the X" → "X"
canon = {}
for k, (t, ic) in rel_stats.items():
    canon_key = k.replace("at the ", "").replace("on the ", "")
    if canon_key not in canon:
        canon[canon_key] = [0, 0]
    canon[canon_key][0] += t
    canon[canon_key][1] += ic

labels  = list(canon.keys())
totals  = [canon[k][0] for k in labels]
incons  = [canon[k][1] for k in labels]
rates   = [100 * ic / t if t > 0 else 0 for ic, t in zip(incons, totals)]

order = np.argsort(rates)[::-1]
labels  = [labels[i] for i in order]
totals  = [totals[i] for i in order]
rates   = [rates[i]  for i in order]

bars = ax.barh(labels, rates, color="#E53935", alpha=0.85)
for i, (b, t, r) in enumerate(zip(bars, totals, rates)):
    ax.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
            f"{r:.1f}% (n={t})", va="center", fontsize=9)
ax.set_xlabel("Inconsistency rate (%)")
ax.set_title("Inconsistency rate by relation label")
ax.set_xlim(0, max(rates) * 1.25 if rates else 1)
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
out_path = "/home/suyang0608/suyang/GroundingDINO/relation_angle_distribution.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nSaved: {out_path}")
