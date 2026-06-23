#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

CUDA_VISIBLE_DEVICES=0 conda run -n dinov3 python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_0.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n dinov3 python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_1.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n dinov3 python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_2.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n dinov3 python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_3.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n dinov3 python train_keypoints.py -c configs/ted_keypoints/dinov3_huge_448x448_lora_r8a16_keypoints_4.yaml
