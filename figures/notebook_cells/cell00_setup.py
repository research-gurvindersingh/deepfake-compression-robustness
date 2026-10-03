# =============================================================================
# CELL 0 — Journal figure setup (run once before the figure cells)
# International Journal of Information Technology (Springer) artwork rules:
#   * width 174 mm (full) or 84 mm (single column), no titles inside figures
#   * sans-serif lettering (Arial/Helvetica) 8-12 pt at printed size
#   * 600 dpi for charts (combination art), vector PDF also saved
# =============================================================================
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import os
# Defaults = Lightning AI studio layout. figures/make_figures.py overrides these
# with environment variables so the same cells also run from the GitHub repo.
BASE_PATH    = Path(os.environ.get("DEEPFAKE_PROJECT", "/teamspace/studios/this_studio/deepfake_project"))
REPORTS_PATH = Path(os.environ.get("DEEPFAKE_REPORTS", BASE_PATH / "reports"))
FIG_OUT      = Path(os.environ.get("DEEPFAKE_FIG_OUT", REPORTS_PATH / "figures_journal"))
FIG_OUT.mkdir(parents=True, exist_ok=True)

FULL_W   = 174 / 25.4      # 6.85 in
SINGLE_W = 84 / 25.4       # 3.31 in

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.title_fontsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "lines.markersize": 4, "grid.linewidth": 0.4, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts
})

METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
MODELS  = ["resnet18", "resnet50", "xception", "efficientnet_b0", "mobilenetv2", "mesonet", "vit"]
SEEDS   = [42, 123, 7]
QUALS   = ["original", "q90", "q70", "q50", "q30"]
QLABELS = ["Original", "Q90", "Q70", "Q50", "Q30"]
LABEL   = {"resnet18": "ResNet18", "resnet50": "ResNet50", "xception": "Xception",
           "efficientnet_b0": "EfficientNet-B0", "mobilenetv2": "MobileNetV2",
           "mesonet": "MesoNet", "vit": "ViT"}
# Okabe-Ito colour-blind-safe palette; markers carry identity in greyscale
COLOR   = {"resnet18": "#0072B2", "resnet50": "#56B4E9", "xception": "#009E73",
           "efficientnet_b0": "#CC79A7", "mobilenetv2": "#E69F00",
           "mesonet": "#999999", "vit": "#D55E00"}
MARKER  = {"resnet18": "o", "resnet50": "s", "xception": "^", "efficientnet_b0": "D",
           "mobilenetv2": "v", "mesonet": "P", "vit": "*"}

# ---- data: single-seed master run (seed 42) and three-seed runs ------------
master = json.load(open(REPORTS_PATH / "multi_method_comparative_results.json"))
ms     = json.load(open(REPORTS_PATH / "multi_seed_results.json"))
ent_path = next(p for p in (REPORTS_PATH / "gradcam_entropy" / "entropy_summary.csv",
                            REPORTS_PATH / "entropy_summary.csv") if p.exists())
ent_df = pd.read_csv(ent_path)

def crs_from(res):
    o = res["original"]["accuracy"]
    return float(np.mean([res[f"q{q}"]["accuracy"] for q in (90, 70, 50, 30)]) / o)

# CRS table (seed 42) — identical to Table 6 / crs_table.csv
crs = pd.DataFrame({m: {d: crs_from(master[d][m]) for d in METHODS} for m in MODELS})

# Entropy vs accuracy-drop pairs (original -> Q30), 35 pairs x 4 cases = 140 rows
paired = ent_df[["Method", "Model", "Case", "entropy_increase_orig_to_q30"]].rename(
    columns={"entropy_increase_orig_to_q30": "entropy_increase"})
paired["accuracy_drop"] = [master[r.Method][r.Model]["original"]["accuracy"]
                           - master[r.Method][r.Model]["q30"]["accuracy"]
                           for r in paired.itertuples()]

def save(fig, n):
    for ext, kw in (("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"})),
                    ("png", dict(dpi=600)), ("pdf", {})):
        fig.savefig(FIG_OUT / f"Fig{n}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    w, h = fig.get_size_inches()
    print(f"Saved Fig{n}.tif/.png/.pdf  ({w*25.4:.0f} x {h*25.4:.0f} mm) -> {FIG_OUT}")
    plt.show()
    plt.close(fig)

print("Setup OK | pairs:", len(paired), "| CRS ViT Deepfakes:", round(crs.loc["Deepfakes", "vit"], 4))
