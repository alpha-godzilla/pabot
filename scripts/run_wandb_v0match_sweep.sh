#!/usr/bin/env bash
set -euo pipefail

cd /home/ljc/code/PaBoT-main
PYTHON_BIN="/home/ljc/miniconda3/envs/formodelling-gpu/bin/python"

LAMBDAS=(0.1 0.3 0.5 0.8)
TS=$(date +%Y%m%d_%H%M%S)

for L in "${LAMBDAS[@]}"; do
  RUN_SUFFIX="v0m${L}_phi0.5"
  RUN_NAME="full_v0distill_rollout_5090_learned_sweep_${TS}/datasets_BtoA/pair1.0_path0.1_vs0.01_ortho0.01_${RUN_SUFFIX}"
  WB_NAME="full_v0distill_rollout_5090_learned_sweep_${TS}_BtoA_${RUN_SUFFIX}"

  echo "============================================================"
  echo "[START] lambda_v0_match=${L}"
  echo "[NAME ] ${RUN_NAME}"
  echo "[W&B  ] ${WB_NAME}"
  echo "============================================================"

  "${PYTHON_BIN}" train.py \
    --dataroot /home/ljc/code/PaBoT-main/datasets \
    --name "${RUN_NAME}" \
    --model dual_velocity_struct \
    --direction BtoA \
    --dataset_mode unaligned \
    --gpu_ids 0 \
    --batch_size 8 \
    --n_epochs 100 \
    --n_epochs_decay 50 \
    --epoch_count 1 \
    --eval_epoch_freq 1 \
    --save_epoch_freq 1 \
    --controlled_pairing True \
    --paired_ratio 0.1 \
    --pair_seed 3407 \
    --a_backbone dit \
    --a_vit_depth 4 \
    --a_vit_dim 256 \
    --a_vit_heads 8 \
    --a_vit_patch 2 \
    --ode_steps 4 \
    --struct_channels 128 \
    --struct_velocity_mode learned \
    --use_structure_attention True \
    --lambda_pair 1.0 \
    --lambda_path 0.1 \
    --lambda_vs 0.01 \
    --lambda_ortho 0.01 \
    --warmup_epochs 10 \
    --structure_attention_source rollout \
    --lambda_v0_match "${L}" \
    --lambda_phi_pair 0.5 \
    --v0_stopgrad_phi True \
    --tag full_v0distill_rollout_5090_learned \
    --no_html \
    --use_wandb \
    --wandb_project PaBoT \
    --wandb_mode online \
    --wandb_run_name "${WB_NAME}"

  echo "[DONE ] lambda_v0_match=${L}"
  echo

done

echo "All sweep runs completed."
