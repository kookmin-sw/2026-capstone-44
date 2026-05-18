#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

log_dir="log/test"
mkdir -p "$log_dir"
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="$log_dir/aerialvg_${timestamp}.log"

method_name="aerialvg"
config_file="$method_name/config/default.py"
latest_checkpoint="$(pick_checkpoint_path "$aerialvg_output_dir/latest.pt" "$aerialvg_checkpoint_path")"
checkpoint_path="$(pick_checkpoint_path "$aerialvg_output_dir/best.pt" "$latest_checkpoint")"
split="test"
image_root="$local_dataset_dir/images"
batch_size=8
num_workers=4
top_k=15

require_local_dataset

device_args=()
hf_token_args=()
[[ -n "$device" ]] && device_args+=(--device "$device")
[[ -n "$hf_token" ]] && hf_token_args+=(--hf-token "$hf_token")

MSDA_DISABLE_EXT="$msda_disable_ext" \
python3 -m "${method_name}.eval" \
  --config-file "$config_file" \
  --checkpoint-path "$checkpoint_path" \
  --dataset-repo "$dataset_repo" \
  --split "$split" \
  --annotation-file "$local_dataset_dir" \
  --image-root "$image_root" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --top-k "$top_k" \
  --log-file "$log_file" \
  "${device_args[@]}" \
  "${hf_token_args[@]}" \
  "$@"
