# =============================================================================
# FIG. 1 — Five-component failure analysis framework (no title in the image)
# =============================================================================
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(FULL_W, 3.1))
ax.set_xlim(0, 18); ax.set_ylim(0, 7.3); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)

def box(x, y, w, h, title, sub="", fc="#f2f2f2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                fc=fc, ec="#333333", lw=0.7))
    if sub:
        ax.text(x + w/2, y + h - 0.12, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(x + w/2, y + (h - 0.45) / 2, sub, ha="center", va="center", fontsize=8,
                color="#333333", linespacing=1.15)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=8.5, fontweight="bold")

box(0.1, 1.5, 3.1, 4.2, "Input", "Trained model\nOriginal test set\nCompressed sets\n(Q90 to Q30)", fc="#e8f0fa")
comps = [("1  Four-case taxonomy", "TP / TN / FP / FN"),
         ("2  CRS", "Compression Robustness Score"),
         ("3  Grad-CAM heatmaps", "At every quality level"),
         ("4  Spatial entropy", "ΔH, original to Q30"),
         ("5  Multi-seed stability", "Seeds 42, 123, 7")]
ys = [5.95, 4.6, 3.25, 1.9, 0.55]
for (t, s), y in zip(comps, ys):
    box(4.2, y, 5.6, 1.1, t, s)
    ax.annotate("", xy=(4.2, y + 0.55), xytext=(3.2, 3.6),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555", mutation_scale=7))
box(11.4, 4.25, 6.3, 2.2, "Signal degradation", "Attention stays on the face;\nthe discriminative signal is lost", fc="#e6f2ff")
box(11.4, 0.95, 6.3, 2.2, "Attention drift", "Attention moves away from\nthe face (MobileNetV2)", fc="#fff0e0")
for y in (ys[0] + 0.55, ys[1] + 0.55):
    ax.annotate("", xy=(11.4, 5.35), xytext=(9.8, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#0072B2", mutation_scale=8))
for y in (ys[2] + 0.55, ys[3] + 0.55, ys[4] + 0.55):
    ax.annotate("", xy=(11.4, 2.05), xytext=(9.8, y), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#D55E00", ls="--", mutation_scale=8))
save(fig, 1)
