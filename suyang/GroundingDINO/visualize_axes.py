"""
Coordinate system diagram: anchor at origin, +x reference, CCW positive angle.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

OUT_PATH = "/home/suyang0608/suyang/GroundingDINO/axes_diagram.png"

fig, ax = plt.subplots(figsize=(9, 9))

L = 1.5  # axis half-length

# Axes (with arrows)
ax.add_patch(FancyArrowPatch((-L, 0), (L, 0), arrowstyle="->",
                             mutation_scale=22, color="black", linewidth=2.2))
ax.add_patch(FancyArrowPatch((0, -L), (0, L), arrowstyle="->",
                             mutation_scale=22, color="black", linewidth=2.2))

# Anchor dot
ax.plot(0, 0, "o", color="black", markersize=10, zorder=5)
ax.text(0.05, -0.12, "anchor", fontsize=13, fontweight="bold", color="black")

# Axis labels
ax.text(L + 0.05, 0.03, "+x",   fontsize=15, fontweight="bold", color="black", va="center")
ax.text(L + 0.05, -0.10, "(right)", fontsize=11, color="#666", va="center")
ax.text(-L - 0.05, 0.03, "-x",   fontsize=15, fontweight="bold", color="black", va="center", ha="right")
ax.text(-L - 0.05, -0.10, "(left)", fontsize=11, color="#666", va="center", ha="right")

ax.text(0.5, L + 0.08, "+y (math) = up",   fontsize=14, fontweight="bold", color="#1976D2")
ax.text(0.5, -L - 0.08, "-y (math) = down", fontsize=14, fontweight="bold", color="#1976D2", va="top")

# Cardinal direction angle labels
ax.text(0.95*L, 0.18, "0°\n(right)",    fontsize=14, fontweight="bold",
        color="#388E3C", ha="center", va="bottom")
ax.text(0.18, 0.95*L, "+90°\n(above)",  fontsize=14, fontweight="bold",
        color="#388E3C", ha="left",   va="top")
ax.text(-0.95*L, 0.18, "180°\n(left)",  fontsize=14, fontweight="bold",
        color="#388E3C", ha="center", va="bottom")
ax.text(-0.95, -0.95*L, "-90°\n(below)", fontsize=14, fontweight="bold",
        color="#388E3C", ha="left",   va="top")

# CCW rotation arc + label
arc = Arc((0, 0), 0.7, 0.7, angle=0, theta1=0, theta2=90,
          color="#E53935", linewidth=2.5)
ax.add_patch(arc)
# arrow head at end of arc (90°)
ax.add_patch(FancyArrowPatch((0.0, 0.35), (-0.02, 0.36), arrowstyle="->",
                             mutation_scale=18, color="#E53935", linewidth=2.5))
ax.text(0.32, 0.32, "CCW = +",
        fontsize=13, fontweight="bold", color="#E53935")

# Reference baseline annotation (highlight +x as 0° reference)
ax.annotate("baseline (+x = 0°)",
            xy=(L, 0), xytext=(0.8, 0.6),
            fontsize=12, color="#1976D2",
            arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.4))

# Title
ax.set_title("Angle measurement reference\n"
             "anchor at origin · +x = 0° baseline · CCW = positive",
             fontsize=14, fontweight="bold", pad=18)

ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.9, 1.9)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
