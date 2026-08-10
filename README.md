# Beyond Accuracy: Compression Robustness in Deepfake Detection

Code and results for the paper *"Beyond Accuracy: A Failure Analysis Framework for Deepfake Detection"*

This repository contains the full training and evaluation pipeline used to produce every table and figure in the paper: the seven-architecture comparison, the Compression Robustness Score (CRS) analysis, the Grad-CAM entropy analysis, the ViT patch-size ablation, and the multi-seed stability checks.

## What's here

```
notebooks/
  main_pipeline.ipynb      Full pipeline: dataset prep, training, evaluation,
                            CRS computation, Grad-CAM/entropy analysis,
                            ViT patch-size ablation, multi-seed runs
reports/
  tables/                  CSV outputs backing the paper's tables
configs/
  paths_example.py         Path variables to adjust for your own environment
requirements.txt
```

Trained model checkpoints and the FaceForensics++ subset itself are not included, checkpoints are available on request, and the dataset must be obtained directly from the original source (see below). Everything needed to regenerate both from scratch is in the notebook.

## Setup

```bash
pip install -r requirements.txt
```

The notebook was developed on a Lightning AI Studio GPU instance. Several path variables are hardcoded to that environment. Before running, copy the variables from `configs/paths_example.py` into the notebook's setup cell and point `BASE_PATH` at wherever you're running this.

## Getting the data

We use FaceForensics++ (Rossler et al., 2019). It is not redistributed here, you need to request access and download it from the official source:

https://github.com/ondyari/FaceForensics

We use five manipulation methods (Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures) at c23 compression, paired with their corresponding real source videos.

### Split protocol

Splits are done at the **source video level**, not the frame level, so that frames from the same source video never appear in more than one split. This matters for deepfake detection specifically, since frames from the same video are highly correlated and a frame-level split can leak information between train and test.

```python
def build_source_splits(common_source_ids, train_ratio=0.7, val_ratio=0.1,
                         test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    source_ids = sorted(list(common_source_ids))
    rng.shuffle(source_ids)
    n = len(source_ids)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    train_ids = source_ids[:train_end]
    val_ids   = source_ids[train_end:val_end]
    test_ids  = source_ids[val_end:]
    return train_ids, val_ids, test_ids
```

Source IDs are split 70/10/20 (train/val/test) with a fixed seed (42 for the main study), then frames are sampled from within each split's source videos:

- 1,400 images per class for training (2,800 total)
- 200 images per class for validation (400 total)
- 400 images per class for testing (800 total)

This is repeated independently for each of the five manipulation methods.

## Reproducing the results

The notebook is organized to run top to bottom. Rough map of what's where:

1. **Dataset preparation** — building per-method datasets, generating JPEG-compressed versions at quality levels 90/70/50/30
2. **Model definitions and training** — seven architectures (ResNet18, ResNet50, Xception, EfficientNet-B0, MobileNetV2, MesoNet, ViT-B/16), two-phase training (classifier head warm-up, then full fine-tune)
3. **Evaluation and CRS computation** — accuracy/F1/ROC-AUC at each compression level, Compression Robustness Score per architecture-method pair
4. **Multi-seed stability runs** — seeds 42, 123, 7 across the main model set
5. **Grad-CAM entropy analysis** — heatmap entropy tracked across compression levels for correctly-classified and misclassified cases (all four confusion categories)
6. **ViT patch-size ablation** — ViT-B/16 vs. ViT-B/8, three seeds, all five methods

All of this is checkpointed: results are saved to JSON after each run and reloaded on rerun, so an interrupted session can be resumed without redoing completed work.

## Requesting checkpoints

Trained model weights are available on request. Open an issue or contact engurvindersingh@gmail.com

## Citation

```
This code accompanies a paper currently under review. A citation entry will be added here once the paper is published.
```

## License

This project is licensed under the MIT License, see the LICENSE file for details.
