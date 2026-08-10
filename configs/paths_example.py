"""
Path configuration template.

The notebook was developed on a Lightning AI Studio instance, so several
path variables are hardcoded to that environment (e.g. /teamspace/studios/...).
Copy this file's variables into the first setup cell of notebooks/main_pipeline.ipynb
and adjust them to your own machine or cloud environment before running.

None of these paths are secrets. This file exists purely so the notebook
is portable outside the original Lightning Studio session it was built in.
"""

from pathlib import Path

# Root folder where the project lives. Everything else is derived from this.
BASE_PATH = Path("/path/to/your/deepfake_project")

# Where the extracted FaceForensics++ subset lives (see README for how to
# obtain and structure this data).
REAL_PATH = BASE_PATH / "dataset" / "real"
FAKE_ROOT_PATH = BASE_PATH / "dataset" / "fake"

# Where model checkpoints get saved during training.
MODELS_PATH = BASE_PATH / "models"

# Where all output tables, JSON results, and entropy/Grad-CAM data get saved.
REPORTS_PATH = BASE_PATH / "reports"
TABLE_OUTPUT_DIR = REPORTS_PATH / "tables"
ENTROPY_DIR = REPORTS_PATH / "gradcam_entropy"

# Aggregate results files referenced throughout the notebook.
MASTER_RESULTS_PATH = REPORTS_PATH / "multi_method_comparative_results.json"
VIT_ABLATION_RESULTS_PATH = REPORTS_PATH / "vit_ablation_results.json"

# If you're running on Google Colab instead of a local machine or Lightning
# Studio, mount Drive first and point BASE_PATH at a folder under
# /content/drive/MyDrive/... instead.
