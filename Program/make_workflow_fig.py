
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Paper", "figures", "workflow.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (title, body lines, border colour) - box heights derive from the line count,
# so no box can ever be too short for its own text.
STAGES = [
    ("1. Terrain preprocessing",
     ["9 DEMNAS tiles: reproject, mosaic,",
      "clip to boundary, fill pits and depressions"], "#2b6cb0"),
    ("2. Conditioning factors",
     ["Elevation, slope, TWI, HAND, dist. to river,",
      "drainage density, rainfall, land cover",
      "(8 factors total)"], "#2f855a"),
    ("3. Spatial block split",
     ["0.05$^{\\circ}$ blocks (~5.5 km)",
      "80% train / 20% test, split by block"], "#b7791f"),
    ("4. Train and evaluate",
     ["Logistic Regression, Random Forest,",
      "XGBoost, LightGBM",
      "F1 | AUC | Acc | P | R | latency"], "#9b2c2c"),
]

TITLE_H, LINE_H, GAP, PAD = 0.46, 0.36, 0.34, 0.17
heights = [TITLE_H + LINE_H * len(b) + 2 * PAD for _, b, _ in STAGES]
total = sum(heights) + GAP * (len(STAGES) - 1)

fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=240)
ax.set_xlim(0, 10)
ax.set_ylim(-0.15, total + 1.15)
ax.axis("off")

ax.text(5.0, total + 0.80, "Flood susceptibility pipeline, Banyumas Regency",
        fontsize=13, fontweight="bold", ha="center", va="center", color="#1a202c")

y = total
for (title, body, colour), h in zip(STAGES, heights):
    ax.add_patch(FancyBboxPatch((0.5, y - h), 9.0, h,
                                boxstyle="round,pad=0.06,rounding_size=0.14",
                                linewidth=1.5, edgecolor=colour,
                                facecolor=colour + "14"))
    ax.text(0.85, y - PAD - TITLE_H / 2, title, fontsize=11.5,
            fontweight="bold", color=colour, ha="left", va="center")
    for k, line in enumerate(body):
        ax.text(0.85, y - PAD - TITLE_H - LINE_H * (k + 0.5), line,
                fontsize=9.4, color="#1a202c", ha="left", va="center")
    y -= h
    if y > GAP:
        ax.add_patch(FancyArrowPatch((5.0, y + 0.02), (5.0, y - GAP + 0.02),
                                     arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.4, color="#4a5568"))
        y -= GAP

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT, "| stage heights:", [round(h, 2) for h in heights])
