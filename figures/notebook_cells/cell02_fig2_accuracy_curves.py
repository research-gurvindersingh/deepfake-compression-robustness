# =============================================================================
# FIG. 2 — Accuracy vs. JPEG quality, all architectures, five methods
#          (mean ± std over seeds 42, 123, 7)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.6), sharey=True, sharex=True)
axf = axes.flatten()
for i, d in enumerate(METHODS):
    ax = axf[i]
    for m in MODELS:
        vals = [[ms[f"{d}__{m}__seed{s}"][q]["accuracy"] for s in SEEDS] for q in QUALS]
        ax.errorbar(range(5), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals],
                    color=COLOR[m], marker=MARKER[m], ms=3.5, lw=1.0, capsize=1.5,
                    elinewidth=0.6, ls=":" if m == "mesonet" else "-")
    ax.axvline(2, color="#777777", ls="--", lw=0.6)
    ax.text(0.03, 0.05, f"({'abcde'[i]}) {d}", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(QLABELS, rotation=0)
    ax.set_ylim(0.40, 1.02); ax.grid(True, ls="--", alpha=0.4)
    if i % 3 == 0: ax.set_ylabel("Accuracy")
axf[5].axis("off")
h = [plt.Line2D([0], [0], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.0, ls=":" if m == "mesonet" else "-",
                label=LABEL[m]) for m in MODELS]
h.append(plt.Line2D([0], [0], color="#777777", ls="--", lw=0.6, label="Q70 (typical platform)"))
axf[5].legend(handles=h, loc="center", title="Architecture", frameon=False)
axf[3].tick_params(labelbottom=True); axf[4].tick_params(labelbottom=True)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
save(fig, 2)
