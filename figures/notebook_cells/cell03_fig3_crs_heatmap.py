# =============================================================================
# FIG. 3 — CRS heatmap (seed 42, same values as Table 6)
#          colour-blind-safe diverging map; values printed in every cell
# =============================================================================
import matplotlib.colors as mcolors
cols = ["vit", "xception", "efficientnet_b0", "resnet18", "resnet50", "mobilenetv2", "mesonet"]
rows = ["FaceShifter", "FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures"]
mat = crs.loc[rows, cols].values
norm = mcolors.TwoSlopeNorm(vmin=0.60, vcenter=0.85, vmax=1.02)
fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
im = ax.imshow(mat, cmap="RdYlBu", norm=norm, aspect="auto")
for i in range(len(rows)):
    for j in range(len(cols)):
        v = mat[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                color="white" if (v < 0.70 or v > 0.97) else "black")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([LABEL[c] + ("*" if c == "mesonet" else "") for c in cols])
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
cb.set_label("CRS"); cb.set_ticks([0.65, 0.75, 0.90, 1.00]); cb.outline.set_linewidth(0.5)
fig.tight_layout()
save(fig, 3)
