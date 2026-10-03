# =============================================================================
# FIG. 9 — NeuralTextures accuracy vs. JPEG quality with platform bands
#          (mean ± std over three seeds)
# =============================================================================
fig, ax = plt.subplots(figsize=(FULL_W, 3.1))
bands = [(1.5, 2.5, "#dbe9f6", "Messaging apps\n(Q70 to Q75)"),
         (2.5, 3.5, "#fff3c4", "Older social\nplatforms (Q50 to Q60)"),
         (3.5, 4.5, "#fde0c5", "Archive / legacy\n(Q30 to Q40)")]
for x0, x1, c, t in bands:
    ax.axvspan(x0, x1, color=c, zorder=0, lw=0)
    ax.text((x0 + x1) / 2, 0.935, t, ha="center", va="top", fontsize=8, color="#333333")
for m in MODELS:
    vals = [[ms[f"NeuralTextures__{m}__seed{s}"][q]["accuracy"] for s in SEEDS] for q in QUALS]
    ax.errorbar(range(5), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals],
                color=COLOR[m], marker=MARKER[m], ms=4, capsize=1.5, elinewidth=0.6,
                lw=1.8 if m == "vit" else 1.0, ls=":" if m == "mesonet" else "-", label=LABEL[m])
ax.axhline(0.5, color="#555555", ls="--", lw=0.7)
ax.text(4.45, 0.493, "Chance (0.50)", ha="right", va="top", fontsize=8, color="#333333")
ax.set_xticks(range(5)); ax.set_xticklabels(QLABELS); ax.set_xlim(-0.4, 4.5)
ax.set_ylim(0.45, 0.95); ax.set_ylabel("Accuracy (NeuralTextures)")
ax.grid(axis="y", ls="--", alpha=0.4)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Architecture")
fig.tight_layout()
save(fig, 9)
