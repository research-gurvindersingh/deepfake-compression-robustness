"""
make_figures.py - regenerate the journal figures from the released result files.

    python figures/make_figures.py                 # Figs 1-5 and 7-10 from results/
    python figures/make_figures.py --with-gradcam --project /path/to/deepfake_project

The figure code lives in figures/notebook_cells/ (one file per notebook cell), so the
same code runs in the Lightning AI notebook and here. Output: figures/output/FigN.{tif,png,pdf}
(600 dpi, 174 mm or 84 mm wide, sans-serif 8-9 pt lettering).
Fig. 6 (Grad-CAM) needs the trained weights, the FaceForensics++ test images and the
helper functions defined in the notebook, so it is only run with --with-gradcam from
inside the notebook session (see README).
"""
import argparse
import os
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
CELLS = sorted((ROOT / "figures" / "notebook_cells").glob("cell*.py"))

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--out", default=str(ROOT / "figures" / "output"))
ap.add_argument("--project", default=str(ROOT))
ap.add_argument("--with-gradcam", action="store_true")
ap.add_argument("--show", action="store_true")
args = ap.parse_args()
if not args.show:
    matplotlib.use("Agg")
os.environ.update(DEEPFAKE_PROJECT=args.project, DEEPFAKE_REPORTS=args.results, DEEPFAKE_FIG_OUT=args.out)

ns = {}
for cell in CELLS:
    if "fig6" in cell.name and not args.with_gradcam:
        print(f"skip {cell.name} (needs trained models; use --with-gradcam in the notebook session)")
        continue
    print(f"run  {cell.name}")
    exec(compile(cell.read_text(), str(cell), "exec"), ns)
