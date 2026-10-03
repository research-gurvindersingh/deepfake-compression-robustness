"""
crs_compute.py - Compression Robustness Score (CRS).

CRS(m, d) = (1/4) * sum_{q in {90, 70, 50, 30}} A(m, d, q) / A(m, d, original)

Usage:
    python src/crs_compute.py --results results/multi_method_comparative_results.json
    python src/crs_compute.py --results results/multi_seed_results.json --multi-seed
"""
import argparse
import json

import numpy as np

QUALITY_LEVELS = [90, 70, 50, 30]


def crs_of(entry, ndigits=4):
    """CRS for one result entry {'original': {...}, 'q90': {...}, ...}; None if undefined.
    ndigits=None returns the unrounded value (used for rank statistics)."""
    orig = entry.get("original", {}).get("accuracy")
    if not orig:
        return None
    accs = [entry[f"q{q}"]["accuracy"] for q in QUALITY_LEVELS if f"q{q}" in entry]
    if not accs:
        return None
    v = float(np.mean(accs) / orig)
    return v if ndigits is None else round(v, ndigits)


def compute_crs(results):
    """results: {method: {model: entry}} (single-seed master file)."""
    return {method: {model: crs_of(entry) for model, entry in models.items()}
            for method, models in results.items()}


def compute_crs_multi_seed(ms):
    """ms: {'Method__model__seedN': entry}. Returns {method: {model: {'mean','std','seeds'}}}."""
    grouped = {}
    for key, entry in ms.items():
        method, model, seed = key.split("__")
        grouped.setdefault(method, {}).setdefault(model, []).append(crs_of(entry))
    return {method: {model: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4),
                             "seeds": v} for model, v in models.items()}
            for method, models in grouped.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--multi-seed", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.results))
    out = compute_crs_multi_seed(data) if args.multi_seed else compute_crs(data)
    print(json.dumps(out, indent=2))
