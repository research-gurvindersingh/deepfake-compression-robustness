# =============================================================================
# FIG. 4 — CRS distribution per architecture across the five methods (seed 42)
# =============================================================================
order = ["vit", "xception", "efficientnet_b0", "resnet18", "resnet50", "mobilenetv2", "mesonet"]
data = [crs[m].values for m in order]
fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), showfliers=False)
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor(COLOR[m]); patch.set_alpha(0.55); patch.set_linewidth(0.7)
rng = np.random.default_rng(42)
for k, (v, m) in enumerate(zip(data, order), start=1):
    ax.scatter(k + rng.normal(0, 0.06, len(v)), v, color=COLOR[m], marker=MARKER[m],
               s=16, edgecolors="black", linewidths=0.3, zorder=3)
ax.axhline(0.90, color="#333333", ls="--", lw=0.7)
ax.axhline(0.75, color="#333333", ls=":", lw=0.8)
ax.text(7.62, 0.90, "Robust\n(≥ 0.90)", va="center", fontsize=8)
ax.text(7.62, 0.75, "Moderate\n(≥ 0.75)", va="center", fontsize=8)
ax.set_xlim(0.4, 8.5)
ax.set_xticks(range(1, 8)); ax.set_xticklabels([LABEL[m] + ("*" if m == "mesonet" else "") for m in order])
ax.set_ylabel("CRS"); ax.set_ylim(0.60, 1.05); ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout()
save(fig, 4)
