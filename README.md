# Beyond Accuracy: A Failure Analysis Framework for Deepfake Detection

Code, results and figure scripts for the paper *Beyond Accuracy: A Failure Analysis Framework for Deepfake Detection* (under review, International Journal of Information Technology).

The study trains seven detectors (ResNet18, ResNet50, Xception, EfficientNet-B0, MobileNetV2, MesoNet, ViT-B/16) on five FaceForensics++ manipulation methods, evaluates them at JPEG quality 90, 70, 50 and 30 with three seeds (105 models, 420 compressed evaluations), and explains the failures with a five-part framework: a four-case failure taxonomy, the Compression Robustness Score (CRS), Grad-CAM, spatial entropy of the heatmaps, and multi-seed stability.

## Repository layout

```
notebooks/deepfake_failure_analysis.ipynb   full pipeline: data prep, training, evaluation,
                                            Grad-CAM, entropy, statistics, journal figures, export
src/crs_compute.py                          Compression Robustness Score
src/entropy_analysis.py                     spatial entropy of a Grad-CAM heatmap
src/reproduce_statistics.py                 recomputes the paper's statistics from results/
figures/notebook_cells/                     one file per figure cell (same code as the notebook)
figures/make_figures.py                     regenerates the figures from results/
figures/output/                             Fig1-Fig10 as PDF (vector) and PNG (600 dpi)
results/                                    released result files (see below)
tools/export_github_release.py              rebuilds this release from a Lightning AI studio
```

## Quick start (no GPU, no dataset)

```bash
pip install numpy pandas scipy matplotlib pillow
python src/reproduce_statistics.py --results-dir results
python figures/make_figures.py
```

`reproduce_statistics.py` prints the CRS table, the entropy correlations and the seed-stability coefficients, and writes them to `results/derived/`. `make_figures.py` rebuilds Figs. 1–5 and 7–10. Fig. 6 needs the trained weights and test images (next section).

## Full reproduction

1. Request access to [FaceForensics++](https://github.com/ondyari/FaceForensics) and extract frames into `raw_dataset/real` and `raw_dataset/fake/<Method>`.
2. `pip install -r requirements.txt`
3. Open `notebooks/deepfake_failure_analysis.ipynb`. Paths default to a Lightning AI studio (`/teamspace/studios/this_studio/deepfake_project`); change `BASE_PATH` in cell 3 for another machine.
4. Run the cells in order:

| Cells | Step |
|---|---|
| 0–7 | installs, paths, hyperparameters, seed |
| 8–10 | identity-disjoint train/val/test split (1,400 / 200 / 400 images per class) and JPEG compression |
| 11–22 | transforms, models, two-phase training, evaluation helpers |
| 23–29 | multi-seed training (seeds 42, 123, 7) for all methods and architectures |
| 30, 67–75 | ViT patch-size ablation (16×16 vs 8×8, three seeds) |
| 31–35 | master results, Grad-CAM, spatial entropy on the full test set |
| 36–45, 59–66 | tables, CRS floor-effect analysis, statistical tests |
| Journal figures | CELL 0, then FIG. 1 to FIG. 10 |
| Export | builds `github_release_final.zip` |

Training used one NVIDIA T4. Cells 49–58 are the earlier draft figures and are kept only for the record. Cell numbers count from 0 in the notebook as shipped.

## Paper figures and tables

| Paper | Source |
|---|---|
| Table 4, Table 5 | `multi_seed_results.json` (three-seed mean) / `multi_method_comparative_results.json` (seed 42) |
| Table 6, Figs. 3, 4, 10 | CRS from `multi_method_comparative_results.json` (seed 42), also `results/crs_table.csv` |
| Table 7, Fig. 5 | false negatives in `multi_method_comparative_results.json` (seed 42) |
| Table 9, Figs. 2, 8, 9 | `multi_seed_results.json` |
| Section 5.7, Fig. 7 | `entropy_summary.csv` + accuracy drop, original to Q30 (n = 140 = 35 pairs × 4 cases) |
| Section 6.2 ablation | `vit_ablation_results.json` |
| Fig. 6 | trained ResNet50 and MobileNetV2 (Deepfakes), FIG. 6 cell |

## Result files

| File | Contents |
|---|---|
| `multi_method_comparative_results.json` | seed-42 run: accuracy, precision, recall, F1, ROC-AUC, FP, FN, confusion matrix for every method × model × quality |
| `multi_seed_results.json` | the same metrics for seeds 42, 123 and 7 (keys `Method__model__seedN`) |
| `entropy_summary.csv` | mean Grad-CAM entropy per method, model and prediction case at each quality level |
| `correlation_statistics.csv` | entropy increase and accuracy drop (original to Q30) per method, model and case |
| `crs_table.csv` | CRS per method and model (seed 42) |
| `vit_ablation_results.json`, `tables/` | ablation results and summary tables (added by the export cell) |

## Notes

- CRS divides by original-quality accuracy. For MesoNet, whose baseline is near chance on three methods, a CRS near 1.0 reflects a performance floor rather than robustness; the paper treats CRS as informative only when baseline AUC exceeds 0.75.
- Model weights and dataset frames are not redistributed. FaceForensics++ is subject to its own terms of use.
