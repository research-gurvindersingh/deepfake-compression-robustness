# =============================================================================
# EXPORT CELL — build the final GitHub release on Lightning AI
# 1. Upload deepfake-compression-robustness.zip into
#    /teamspace/studios/this_studio/deepfake_project/
# 2. Run the journal figure cells (CELL 0 to FIG. 10) so reports/figures_journal/ exists
# 3. Run this cell. Output: deepfake_project/github_release_final.zip
# =============================================================================
import shutil, subprocess, sys, zipfile
from pathlib import Path

BASE_PATH    = Path("/teamspace/studios/this_studio/deepfake_project")
REPORTS_PATH = BASE_PATH / "reports"
TABLES       = REPORTS_PATH / "tables"
SKELETON_ZIP = BASE_PATH / "deepfake-compression-robustness.zip"
WORK         = BASE_PATH / "github_release_final"
REPO         = WORK / "deepfake-compression-robustness"
OUT_ZIP      = BASE_PATH / "github_release_final.zip"

if not SKELETON_ZIP.exists():
    raise FileNotFoundError(f"Upload deepfake-compression-robustness.zip to {BASE_PATH} first.")
shutil.rmtree(WORK, ignore_errors=True)
with zipfile.ZipFile(SKELETON_ZIP) as z:
    z.extractall(WORK)

# results produced on the studio (newest copies overwrite the packaged ones)
files = {
    REPORTS_PATH / "multi_method_comparative_results.json": "results",
    REPORTS_PATH / "multi_seed_results.json": "results",
    REPORTS_PATH / "vit_ablation_results.json": "results",
    REPORTS_PATH / "gradcam_entropy" / "entropy_summary.csv": "results",
    TABLES / "crs_table.csv": "results",
    TABLES / "correlation_statistics.csv": "results",
    TABLES / "architecture_statistical_tests.csv": "results/tables",
    TABLES / "entropy_by_case_type.csv": "results/tables",
    TABLES / "vit_patch_ablation_table.csv": "results/tables",
    TABLES / "crs_redesign_table.csv": "results/tables",
    TABLES / "summary_degradation_table.csv": "results/tables",
    TABLES / "accuracy_table.csv": "results/tables",
    TABLES / "f1_table.csv": "results/tables",
    TABLES / "roc_auc_table.csv": "results/tables",
    TABLES / "false_positives_table.csv": "results/tables",
    TABLES / "false_negatives_table.csv": "results/tables",
}
for src, dst in files.items():
    if src.exists():
        (REPO / dst).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, REPO / dst / src.name)
        print("added   ", f"{dst}/{src.name}")
    else:
        print("missing ", src.relative_to(BASE_PATH), "(skipped)")

# journal figures, including Fig. 6 generated from the trained models (TIFFs stay out of git)
fig_src = REPORTS_PATH / "figures_journal"
for f in sorted(fig_src.glob("Fig*.p*")):          # .png and .pdf
    shutil.copy2(f, REPO / "figures" / "output" / f.name)
print("figures ", sorted(p.name for p in (REPO / "figures" / "output").glob("Fig6.*")) or "Fig6 not found: run the FIG. 6 cell")

# refresh derived statistics from the copied results
subprocess.run([sys.executable, str(REPO / "src" / "reproduce_statistics.py"),
                "--results-dir", str(REPO / "results")], check=True, cwd=REPO / "src")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()
shutil.make_archive(str(OUT_ZIP.with_suffix("")), "zip", WORK, "deepfake-compression-robustness")
print("\nRelease ready:", OUT_ZIP, f"({OUT_ZIP.stat().st_size / 1e6:.1f} MB)")
