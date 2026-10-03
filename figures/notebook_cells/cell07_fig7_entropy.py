# =============================================================================
# FIG. 7 — Entropy increase vs. accuracy drop, ORIGINAL -> Q30
# Same data as the statistics in the text (n = 140 = 35 pairs x 4 cases):
#   (a) six signal-degradation architectures, n = 120
#   (b) MobileNetV2, n = 20  -> Spearman rho = 0.552, p = 0.012
# (The old cell pooled Q90-Q30 for panel b, which gave rho = 0.71 and did not
#  match the paper.)
# =============================================================================
MCOL = dict(zip(METHODS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
MMRK = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
a = paired[paired.Model != "mobilenetv2"]
b = paired[paired.Model == "mobilenetv2"]
ra, pa = pearsonr(a.entropy_increase, a.accuracy_drop)
rb, pb = spearmanr(b.entropy_increase, b.accuracy_drop)

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.9))
for m in [x for x in MODELS if x != "mobilenetv2"]:
    g = a[a.Model == m]
    axes[0].scatter(g.entropy_increase, g.accuracy_drop, s=14, color=COLOR[m], marker=MARKER[m],
                    edgecolors="black", linewidths=0.25, label=LABEL[m])
for d in METHODS:
    g = b[b.Method == d]
    axes[1].scatter(g.entropy_increase, g.accuracy_drop, s=20, color=MCOL[d], marker=MMRK[d],
                    edgecolors="black", linewidths=0.25, label=d)
for ax, df, txt, tag, pos in ((axes[0], a, f"Pearson r = {ra:.3f}, p = {pa:.3f}\nn = {len(a)}", "(a)", (0.03, 0.03, "bottom")),
                              (axes[1], b, f"Spearman ρ = {rb:.3f}, p = {pb:.3f}\nn = {len(b)}", "(b)", (0.03, 0.97, "top"))):
    k, c0 = np.polyfit(df.entropy_increase, df.accuracy_drop, 1)
    xs = np.linspace(df.entropy_increase.min(), df.entropy_increase.max(), 50)
    ax.plot(xs, k * xs + c0, "k--", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.text(pos[0], pos[1], f"{tag}  {txt}", transform=ax.transAxes, va=pos[2], fontsize=8,
            bbox=dict(fc="white", ec="#999999", lw=0.4, pad=2))
    ax.set_xlabel("Entropy increase ΔH (original to Q30)")
    ax.grid(True, ls="--", alpha=0.4)
axes[0].set_ylabel("Accuracy drop (original to Q30)")
axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, ncol=2, handletextpad=0.2, columnspacing=0.6)
axes[1].legend(loc="lower right", frameon=True, framealpha=0.9, handletextpad=0.2)
for ax in axes: ax.set_ylim(-0.05, 0.62)
fig.tight_layout(w_pad=1.0)
save(fig, 7)
print(f"(a) r={ra:.4f} p={pa:.4f} | (b) rho={rb:.4f} p={pb:.4f}")
