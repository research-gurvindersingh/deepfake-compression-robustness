# =============================================================================
# FIG. 10 — CRS per manipulation method, four architectures (single column)
# =============================================================================
axes_m = ["FaceShifter", "FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures"]
archs  = ["vit", "xception", "resnet18", "mobilenetv2"]
ang = np.linspace(0, 2 * np.pi, len(axes_m), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(SINGLE_W, 3.5), subplot_kw=dict(polar=True))
for m in archs:
    v = [crs.loc[d, m] for d in axes_m]; v += v[:1]
    ax.plot(ang, v, color=COLOR[m], marker=MARKER[m], ms=4, lw=1.1, label=LABEL[m])
    ax.fill(ang, v, color=COLOR[m], alpha=0.08)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(th, [0.90] * 200, color="#333333", ls="--", lw=0.6, label="Robust threshold (0.90)")
ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_m)
ax.set_ylim(0.55, 1.02); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=8); ax.set_rlabel_position(108)
ax.tick_params(axis="x", pad=7); ax.grid(lw=0.4)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8)
fig.tight_layout()
save(fig, 10)
