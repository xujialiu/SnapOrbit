# SnapOrbit

Code for *"Uncertainty-Aware Deep Learning Framework for Orbital Disease Triage
Using Smartphone-Captured Images: A Prospective Study."*

[English](README.md) | [中文](README.zh.md)

A two-stage **DINOv3 + LoRA** pipeline that classifies **TED / SOLs / normal**
from smartphone external eye photos:

1. **Keypoint localization** — detect the 4 canthal landmarks, then crop a
   **bilateral** (448×896) or **unilateral** (448×448) ROI.
2. **Ternary classifier** — DINOv3-Huge + LoRA, with **snapshot-ensemble
   uncertainty** to defer the most uncertain cases.

> No patient data or trained weights are included. `projects/` is git-ignored
> and holds runtime outputs and your local data.

## Setup

```bash
conda create -n ted python=3.11 -y && conda activate ted
# install PyTorch for your CUDA (https://pytorch.org), then:
pip install -r requirements.txt
```

DINOv3 weights are downloaded automatically by `timm` (≥ 1.0.19) on first use.

## Data

Prepare the CSVs below (not distributed) with a patient-level `fold` column
(0–9) for cross-validation, and set `dataset.image_path` / `csv_path` in the
configs.

- **Classification** — long format, 4 canthus rows per image:
  `relative_path, keypoint_x, keypoint_y, label, fold` (`label`: 0=normal,
  1=TED, 2=SOLs; keypoints in percent 0–100).
- **Keypoints** — wide format, one row per image, coords normalized 0–1:
  `relative_path, x1,y1, x2,y2, x3,y3, x4,y4, fold`. Build it with
  `python data_ted/codes/make_keypoints_csv.py`.

## Usage

Run from the repo root. Configs live in `configs/`; any field can be overridden
on the CLI (e.g. `device=cuda:0`).

```bash
# 1) Keypoint model
python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_0.yaml

# 2) Classifier — bilateral (primary) / unilateral (ablation)
python train_classification.py -c configs/ted_classification/dinov3_huge_448x896_lora_r8a16_d01_dp01_f1_aug_v3_double_weighted-sampler_0.yaml
#    evaluate: add  --eval paths.ckpt_path=<checkpoint.pth>

# 3) Uncertainty (snapshot ensemble = 6 best-val-F1 ckpts × 5 folds; predictive entropy)
python uncertainty_classification.py -c configs/ted_classification/<run>/config_eval_snap.yaml

# 4) Explainability (ViT attention maps)
python explainability_classification.py -c configs/ted_classification/<run>/config_exp.yaml
```

5-fold helper scripts: `configs/ted_keypoints/script_train_5fold.sh`,
`configs/ted_classification/script_train_{double,single}.sh`, and
`script_{uncertainty,explainability,reader_study}.sh`.

## Layout

```
SnapOrbit/
├── train_classification.py           # entry point: TED / SOLs / normal classifier
├── train_keypoints.py                # entry point: canthal keypoint model
├── engine_finetune_*.py              # train / eval loops for the two entry points
├── uncertainty_classification.py     # uncertainty: snapshot_ensemble | mc_dropout | mc_dropout_snapshot_ensemble
├── explainability_classification.py  # ViT attention maps
├── mc_dropout.py                     # MCDropout wrapper used by the mc_dropout methods above
├── models/                           # ViT backbone + LoRA wrapper
├── datasets/                         # classification / keypoints datasets
├── augmentations/                    # classification / keypoints augmentations
├── loss/                             # cross-entropy (classification), MSE (keypoints)
├── util/                             # config parsing, LR schedule, training utils
├── configs/
│   ├── ted_classification/           # classifier configs + script_*.sh (train / uncertainty / explainability / reader study)
│   └── ted_keypoints/                # keypoint configs + script_*.sh (train / 5-fold / eval-vis)
├── data_ted/codes/                   # data prep (make_keypoints_csv.py) & figure scripts
├── visualization/                    # keypoint overlay visualization
└── projects/                         # git-ignored: runtime outputs & local data
```

## Key settings

DINOv3-Huge+ backbone with LoRA on the attention QKV projection (bilateral
r8/α16, unilateral r32/α64, dropout 0.1); input 448×896 / 448×448;
cross-entropy with a class-weighted sampler; base LR 1e-3, weight decay 0.05,
100 epochs (cosine + 10-epoch warmup), mixed precision. See `configs/` for the
exact per-experiment hyperparameters.

## License

Released under the [MIT License](LICENSE).
