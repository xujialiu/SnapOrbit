#!/usr/bin/env bash
# Test-eval + keypoint visualization for all 5 folds, using each fold's lowest-MSE
# checkpoint (set in the per-fold config_test.yaml / config_vis_kp.yaml).
# Set CUDA_VISIBLE_DEVICES to pick a GPU.  Outputs:
#   eval  -> projects/ted_keypoints/<run>/test/metrics.csv
#   vis   -> projects/ted_keypoints/<run>/vis_kp/
set -e
cd "$(dirname "$0")/../.."
STEM=dinov3_huge_448x448_lora_r8a16_keypoints

for i in 0 1 2 3 4; do
  echo "===== fold $i : test eval ====="
  conda run -n dinov3 python train_keypoints.py \
    -c configs/ted_keypoints/${STEM}_${i}/config_test.yaml
  echo "===== fold $i : visualize ====="
  conda run -n dinov3 python visualization/visualize_keypoint.py \
    -c configs/ted_keypoints/${STEM}_${i}/config_vis_kp.yaml
done
echo "ALL 5 FOLDS DONE"
