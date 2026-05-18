"""
Analyze per-sample eval results: failure categorization, breakdowns.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

PER_SAMPLE = "/home/suyang0608/suyang/GroundingDINO/per_sample_srbm_bce_l3.jsonl"
OUT_DIR    = "/home/suyang0608/suyang/GroundingDINO/failure_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

with open(PER_SAMPLE) as f:
    records = [json.loads(l) for l in f if l.strip()]

N = len(records)
top1 = sum(r["top1_correct"] for r in records) / N
top5 = sum(r["top5_correct"] for r in records) / N
print(f"Total samples: {N}")
print(f"Top-1: {top1*100:.2f}%   Top-5: {top5*100:.2f}%")

# ── (A) Top-1 / Top-5 failure split ──────────────────────────────────────────
def categorize(r):
    if r["top1_correct"]: return "top1_correct"
    elif r["top5_correct"]: return "top5_only"   # top-1 miss but top-5 hit
    else: return "all_miss"

cat_counts = Counter(categorize(r) for r in records)
print("\n=== Failure category split ===")
for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:15s}: {v} ({100*v/N:.1f}%)")

# ── (B) Anchor size breakdown ────────────────────────────────────────────────
print("\n=== Top-1 accuracy by anchor size bin ===")
size_bins = [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.0)]
labels = ["tiny\n(<0.5%)", "small\n(0.5-2%)", "med-s\n(2-5%)", "med-l\n(5-10%)", "large\n(10-30%)", "huge\n(>30%)"]
size_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
for r in records:
    a = r["anchor_area"]
    for i, (lo, hi) in enumerate(size_bins):
        if lo <= a < hi:
            size_stats[i]["total"] += 1
            size_stats[i]["top1"]  += r["top1_correct"]
            size_stats[i]["top5"]  += r["top5_correct"]
            break

print(f"{'bin':<15} {'N':>6}  {'Top-1':>7}  {'Top-5':>7}")
for i, lbl in enumerate(labels):
    s = size_stats[i]
    if s["total"] == 0: continue
    t1 = s["top1"]/s["total"]*100
    t5 = s["top5"]/s["total"]*100
    print(f"{lbl[:15]:<15} {s['total']:>6}  {t1:>6.1f}%  {t5:>6.1f}%")

# ── (C) Relation count breakdown ─────────────────────────────────────────────
print("\n=== Top-1 by number of references ===")
nref_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
for r in records:
    n = r["num_refs"]
    nref_stats[n]["total"] += 1
    nref_stats[n]["top1"]  += r["top1_correct"]
    nref_stats[n]["top5"]  += r["top5_correct"]

for n in sorted(nref_stats.keys()):
    s = nref_stats[n]
    t1 = s["top1"]/s["total"]*100
    t5 = s["top5"]/s["total"]*100
    print(f"  refs={n}  N={s['total']:>5}  Top-1={t1:.1f}%  Top-5={t5:.1f}%")

# ── (D) Relation type breakdown ──────────────────────────────────────────────
print("\n=== Top-1 by relation type (anchor has THIS relation in some ref) ===")
rel_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
def canon_rel(rel):
    rel = rel.lower().strip()
    rel = rel.replace("at the ", "").replace("on the ", "")
    return rel

for r in records:
    rels_canon = set(canon_rel(x) for x in r["relations"])
    for rel in rels_canon:
        rel_stats[rel]["total"] += 1
        rel_stats[rel]["top1"]  += r["top1_correct"]
        rel_stats[rel]["top5"]  += r["top5_correct"]

ordered = sorted(rel_stats.items(), key=lambda x: x[1]["total"], reverse=True)
print(f"{'relation':<22} {'N':>6}  {'Top-1':>7}  {'Top-5':>7}")
for rel, s in ordered:
    t1 = s["top1"]/s["total"]*100
    t5 = s["top5"]/s["total"]*100
    print(f"{rel:<22} {s['total']:>6}  {t1:>6.1f}%  {t5:>6.1f}%")

# ── (E) Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (E1) Failure pie
ax = axes[0]
labels_pie = []
sizes = []
colors = ["#43A047", "#FB8C00", "#E53935"]
for k, lbl in [("top1_correct", "Top-1 correct"),
               ("top5_only", "Top-5 only\n(rank issue)"),
               ("all_miss", "All miss\n(recall issue)")]:
    v = cat_counts.get(k, 0)
    labels_pie.append(f"{lbl}\n{v} ({100*v/N:.1f}%)")
    sizes.append(v)
ax.pie(sizes, labels=labels_pie, colors=colors, autopct="", startangle=90)
ax.set_title("Failure category split")

# (E2) Top-1 by anchor size
ax = axes[1]
xs = list(range(len(size_bins)))
t1s = []; ns = []
for i in range(len(size_bins)):
    s = size_stats[i]
    if s["total"] > 0:
        t1s.append(s["top1"]/s["total"]*100); ns.append(s["total"])
    else:
        t1s.append(0); ns.append(0)
bars = ax.bar(xs, t1s, color="#1976D2")
for b, n in zip(bars, ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"n={n}",
            ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Top-1 by anchor area")
ax.set_ylim(0, max(t1s)*1.2 if t1s else 100)
ax.grid(alpha=0.3, axis="y")

# (E3) Top-1 by relation type
ax = axes[2]
relabel = [r for r, _ in ordered]
t1_vals = [s["top1"]/s["total"]*100 for _, s in ordered]
n_vals  = [s["total"] for _, s in ordered]
bars = ax.barh(relabel, t1_vals, color="#7E57C2")
for b, n in zip(bars, n_vals):
    ax.text(b.get_width()+1, b.get_y()+b.get_height()/2,
            f"n={n}", va="center", fontsize=8)
ax.set_xlabel("Top-1 accuracy (%)")
ax.set_title("Top-1 by relation type")
ax.invert_yaxis()
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
fig.suptitle(f"Failure analysis — GDINO+LoRA+SRBM (BCE L=3) — Top-1 {top1*100:.2f}%",
             y=1.02, fontsize=13, fontweight="bold")
fig_path = f"{OUT_DIR}/breakdown.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
print(f"\nSaved: {fig_path}")
