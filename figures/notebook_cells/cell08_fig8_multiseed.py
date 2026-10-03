# =============================================================================
# FIG. 8 — CRS rank stability across seeds (35 architecture-method pairs)
# =============================================================================
crs_seed = {s: np.array([crs_from(ms[f"{d}__{m}__seed{s}"]) for d in METHODS for m in MODELS]) for s in SEEDS}
lab_m = [m for d in METHODS for m in MODELS]
rank = {s: crs_seed[s].argsort().argsort() + 1 for s in SEEDS}
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))
for ax, (sx, sy), tag in zip(axes, [(42, 123), (42, 7)], "ab"):
    rho, p = spearmanr(crs_seed[sx], crs_seed[sy])
    for i, m in enumerate(lab_m):
        ax.scatter(rank[sx][i], rank[sy][i], s=18, color=COLOR[m], marker=MARKER[m],
                   edgecolors="black", linewidths=0.25)
    ax.plot([1, 35], [1, 35], "k--", lw=0.7)
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.03, 0.97, f"({tag})  ρ = {rho:.4f}, {ptxt}", transform=ax.transAxes, va="top",
            bbox=dict(fc="white", ec="#999999", lw=0.4, pad=2))
    ax.set_xlabel(f"CRS rank, seed {sx}"); ax.set_ylabel(f"CRS rank, seed {sy}")
    ax.set_xlim(0, 36); ax.set_ylim(0, 36); ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.4)
h = [plt.Line2D([0], [0], ls="", marker=MARKER[m], color=COLOR[m], mec="black", mew=0.25, ms=5, label=LABEL[m]) for m in MODELS]
fig.legend(handles=h, loc="center right", frameon=False, title="Architecture")
fig.tight_layout(rect=(0, 0, 0.80, 1))
save(fig, 8)
