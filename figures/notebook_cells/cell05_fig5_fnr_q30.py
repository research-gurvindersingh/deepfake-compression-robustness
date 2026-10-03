# =============================================================================
# FIG. 5 — False negative rate at Q30 (FN / 400 test fakes), seed 42 = Table 7
# =============================================================================
x = np.arange(len(METHODS)); w = 0.115
off = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * w
HATCH = {"mesonet": "////", "mobilenetv2": "....", "vit": "xxxx"}
fig, ax = plt.subplots(figsize=(FULL_W, 2.9))
for j, m in enumerate(MODELS):
    fnr = [master[d][m]["q30"]["false_negatives"] / 400 for d in METHODS]
    ax.bar(x + off[j], fnr, width=w * 0.92, color=COLOR[m], hatch=HATCH.get(m, ""),
           edgecolor="black", linewidth=0.3, label=LABEL[m])
ax.axhline(0.50, color="#333333", ls="--", lw=0.7, label="50% miss rate")
ax.set_xticks(x); ax.set_xticklabels(METHODS)
ax.set_ylabel("False negative rate at Q30"); ax.set_ylim(0, 1.05)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.legend(ncol=4, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.22))
fig.tight_layout()
save(fig, 5)
