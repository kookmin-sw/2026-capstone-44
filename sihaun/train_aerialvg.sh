#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

log_dir="log/train"
mkdir -p "$log_dir"
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="$log_dir/aerialvg_${timestamp}.log"

method_name="aerialvg"
config_file="$method_name/config/default.py"
# Saves to: /data2/2026_capstone/sihaun/aerialvg/outputs/$checkpoint_dir_name
checkpoint_dir_name="run"
train_split="train"
eval_split="val"
annotation_source="$local_dataset_dir"
image_root="$local_dataset_dir/images"
init_checkpoint="$aerialvg_checkpoint_path"
output_dir="$aerialvg_outputs_dir/$checkpoint_dir_name"

epochs=15
batch_size=8
num_workers=8
lr=0.001
weight_decay=1e-5
lr_drop_epoch=4
grad_clip_norm=0.1
cls_loss_weight=2.0
bbox_loss_weight=5.0
giou_loss_weight=2.0
top_k=15

require_local_dataset

device_args=()
hf_token_args=()
[[ -n "$device" ]] && device_args+=(--device "$device")
[[ -n "$hf_token" ]] && hf_token_args+=(--hf-token "$hf_token")

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MSDA_DISABLE_EXT="$msda_disable_ext" \
python3 -m "${method_name}.train" \
  --config-file "$config_file" \
  --init-checkpoint "$init_checkpoint" \
  --dataset-repo "$dataset_repo" \
  --train-split "$train_split" \
  --eval-split "$eval_split" \
  --annotation-file "$annotation_source" \
  --image-root "$image_root" \
  --output-dir "$output_dir" \
  --log-file "$log_file" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --lr "$lr" \
  --weight-decay "$weight_decay" \
  --lr-drop-epoch "$lr_drop_epoch" \
  --grad-clip-norm "$grad_clip_norm" \
  --cls-loss-weight "$cls_loss_weight" \
  --bbox-loss-weight "$bbox_loss_weight" \
  --giou-loss-weight "$giou_loss_weight" \
  --top-k "$top_k" \
  "${device_args[@]}" \
  "${hf_token_args[@]}" \
  "$@"
