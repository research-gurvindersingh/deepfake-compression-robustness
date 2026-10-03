"""
reproduce_statistics.py - recompute the statistics reported in the paper from the
released result files (no GPU, no dataset needed).

    python src/reproduce_statistics.py --results-dir results

Prints and writes to results/derived/:
  * CRS per architecture and method (seed 42, Table 6) and three-seed mean/std
  * entropy increase vs. accuracy drop, original -> Q30 (n = 140 = 35 pairs x 4 cases):
    overall Pearson / Spearman and per-architecture Spearman (Section 5.7, Fig. 7)
  * CRS rank stability across seeds (Section 5.8, Fig. 8)
  * false negatives at original quality and Q30 (Table 7)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from crs_compute import compute_crs, crs_of

METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
MODELS = ["resnet18", "resnet50", "xception", "efficientnet_b0", "mobilenetv2", "mesonet", "vit"]
SEEDS = [42, 123, 7]


def main(results_dir: Path):
    out = results_dir / "derived"
    out.mkdir(exist_ok=True)
    master = json.load(open(results_dir / "multi_method_comparative_results.json"))
    ms = json.load(open(results_dir / "multi_seed_results.json"))
    ent = pd.read_csv(results_dir / "entropy_summary.csv")

    # --- CRS ----------------------------------------------------------------
    crs42 = pd.DataFrame(compute_crs(master)).T.loc[METHODS, MODELS]
    crs42.to_csv(out / "crs_seed42.csv")
    seed_crs = {s: pd.DataFrame({m: {d: crs_of(ms[f"{d}__{m}__seed{s}"], ndigits=None) for d in METHODS} for m in MODELS})
                for s in SEEDS}
    stack = np.stack([seed_crs[s].values for s in SEEDS])
    pd.DataFrame(stack.mean(0), index=METHODS, columns=MODELS).round(4).to_csv(out / "crs_3seed_mean.csv")
    pd.DataFrame(stack.std(0), index=METHODS, columns=MODELS).round(4).to_csv(out / "crs_3seed_std.csv")
    print("CRS, seed 42 (Table 6):\n", crs42.round(3).to_string(), "\n")

    # --- entropy vs accuracy drop (original -> Q30) -------------------------
    pairs = ent[["Method", "Model", "Case", "N", "entropy_increase_orig_to_q30"]].rename(
        columns={"entropy_increase_orig_to_q30": "entropy_increase"})
    pairs["accuracy_drop"] = [master[r.Method][r.Model]["original"]["accuracy"]
                              - master[r.Method][r.Model]["q30"]["accuracy"] for r in pairs.itertuples()]
    pairs.to_csv(out / "entropy_vs_accuracy_drop.csv", index=False)
    r, p = pearsonr(pairs.entropy_increase, pairs.accuracy_drop)
    rho, ps = spearmanr(pairs.entropy_increase, pairs.accuracy_drop)
    print(f"Entropy vs accuracy drop, all (n={len(pairs)}): Pearson r={r:.3f} (p={p:.3f}); "
          f"Spearman rho={rho:.3f} (p={ps:.2g})")
    rows = []
    for m in MODELS:
        g = pairs[pairs.Model == m]
        rr, pp = spearmanr(g.entropy_increase, g.accuracy_drop)
        rows.append({"Model": m, "n": len(g), "spearman_rho": round(rr, 4), "p_value": round(pp, 4)})
    per_model = pd.DataFrame(rows)
    per_model.to_csv(out / "entropy_correlation_per_model.csv", index=False)
    print(per_model.to_string(index=False), "\n")

    # --- seed stability -------------------------------------------------------
    vec = {s: seed_crs[s].values.ravel() for s in SEEDS}
    for a, b in [(42, 123), (42, 7), (123, 7)]:
        rr, pp = spearmanr(vec[a], vec[b])
        print(f"CRS rank stability seed {a} vs {b}: rho={rr:.4f}, p={pp:.2g}")

    # --- false negatives (Table 7) -------------------------------------------
    fn = pd.DataFrame([{"Method": d, "Model": m,
                        "FN_original": master[d][m]["original"]["false_negatives"],
                        "FN_Q30": master[d][m]["q30"]["false_negatives"]}
                       for d in METHODS for m in MODELS])
    fn.to_csv(out / "false_negatives_table.csv", index=False)
    print(f"\nFalse negatives written ({len(fn)} rows). Derived files in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", type=Path)
    main(ap.parse_args().results_dir)
