"""
8-way compass rose diagram showing 45° windows for each direction.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

OUT_PATH = "/home/suyang0608/suyang/GroundingDINO/compass_diagram.png"

# Direction centers (math convention: 0°=right, 90°=above, increases CCW)
DIRECTIONS = [
    ("right",        0),
    ("top right",   45),
    ("above",       90),
    ("top left",   135),
    ("left",       180),
    ("bottom left",-135),
    ("below",      -90),
    ("bottom right",-45),
]

GREEN = "#43A047"
RED   = "#E53935"
GREEN_FILL = "#43A04722"
RED_FILL   = "#E5393522"

fig, ax = plt.subplots(figsize=(9, 9))

# Outer dashed boundary circle
theta = np.linspace(0, 2*np.pi, 360)
ax.plot(np.cos(theta), np.sin(theta), color="#888", lw=0.7, ls="--")

# 45° wedges centered on each direction (consistent zones, green)
for label, ctr in DIRECTIONS:
    start = ctr - 22.5
    end   = ctr + 22.5
    w = Wedge((0, 0), 1.0, start, end,
              width=0.18, facecolor=GREEN_FILL, edgecolor=GREEN, linewidth=2)
    ax.add_patch(w)

# Boundary lines between sectors (light)
for ctr in [22.5, 67.5, 112.5, 157.5, -157.5, -112.5, -67.5, -22.5]:
    a = np.radians(ctr)
    ax.plot([0, 0.95*np.cos(a)], [0, 0.95*np.sin(a)],
            color="#bbb", lw=0.6, ls="-")

# Center dot
ax.plot(0, 0, "o", color="black", markersize=6)

# Labels (direction name + angle)
for label, ctr in DIRECTIONS:
    a = np.radians(ctr)
    r = 1.18
    x, y = r*np.cos(a), r*np.sin(a)
    ax.text(x, y+0.04, label, ha="center", va="center",
            fontsize=15, fontweight="bold")
    ax.text(x, y-0.06, f"{ctr}°", ha="center", va="center",
            fontsize=12, color="#444")

# Title (English to avoid font issues)
ax.set_title("Inconsistency criterion: 8-way × 45° windows\n"
             "non-overlapping 360° partition over 8 label directions",
             fontsize=14, fontweight="bold", pad=20)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=140, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
