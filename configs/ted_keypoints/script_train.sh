#!/usr/bin/env bash
# TED 4-keypoint detection — DINOv3 Huge+ @ 448x448 LoRA.
set -e
cd "$(dirname "$0")/../.."

# 1) build the wide-format keypoint CSV (run once)
conda run -n dinov3 python data_ted/codes/make_keypoints_csv.py

# 2) train
CUDA_VISIBLE_DEVICES=1 python train_keypoints.py \
  -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints.yaml


python visualization/visualize_keypoint.py \
    -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints/config_vis_kp.yaml