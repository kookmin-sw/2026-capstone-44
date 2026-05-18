#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

log_dir="log/train"
mkdir -p "$log_dir"
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file="$log_dir/method6_${timestamp}.log"

config_file="method6/config/default.py"
# Saves to: /data2/2026_capstone/sihaun/method6/outputs/$checkpoint_dir_name
checkpoint_dir_name="run"
role_gate_mode="learned"
train_split="train"
eval_split="val"
annotation_source="$local_dataset_dir"
image_root="$local_dataset_dir/images"
init_checkpoint="$aerialvg_checkpoint_path"
output_dir="$method6_outputs_dir/$checkpoint_dir_name"

epochs=15
batch_size=8
num_workers=2
lr=0.001
weight_decay=1e-5
lr_drop_epoch=4
grad_clip_norm=0.1
det_loss_weight=1.0
cls_loss_weight=2.0
bbox_loss_weight=5.0
giou_loss_weight=2.0
tau_gt=0.5
tau_aux=0.6
role_loss_weight=0.5
aux_loss_weight=0.0
role_gt_weight=2.0
role_aux_weight=2.0
role_none_weight=0.2

# method6:
# The AerialVG detection/proposal module is frozen.
# stage1 trains only the Role Gate submodule.
# stage2 trains only the proposed role_evidence_module.
stage1_epochs=3
stage2_epochs=12
stage1_det_weight=0.0
stage1_role_weight=1.0
stage1_aux_weight=0.0
stage2_det_weight=1.0
stage2_role_weight=0.1
stage2_aux_weight=0.5
stage2_warmup_ratio=0.1
single_stage=0
top_k=15

stage1_trainable_pattern=(role_text_attn role_context_norm role_mlp)
stage2_trainable_pattern=(role_evidence_module)

disable_checkpoint="${disable_checkpoint:-${DISABLE_CHECKPOINT:-0}}"
disable_transformer_ckpt="${disable_transformer_ckpt:-${DISABLE_TRANSFORMER_CKPT:-0}}"
cuda_sync_debug="${cuda_sync_debug:-${CUDA_SYNC_DEBUG:-0}}"

require_local_dataset

extra_stage_args=()
device_args=()
hf_token_args=()
debug_args=()
stage1_pattern_args=(--stage1-trainable-pattern "${stage1_trainable_pattern[@]}")
stage2_pattern_args=(--stage2-trainable-pattern "${stage2_trainable_pattern[@]}")
[[ "$single_stage" != "0" ]] && extra_stage_args+=(--single-stage)
[[ -n "$stage1_epochs" ]] && extra_stage_args+=(--stage1-epochs "$stage1_epochs")
[[ -n "$stage2_epochs" ]] && extra_stage_args+=(--stage2-epochs "$stage2_epochs")
[[ -n "$device" ]] && device_args+=(--device "$device")
[[ -n "$hf_token" ]] && hf_token_args+=(--hf-token "$hf_token")
[[ "$disable_checkpoint" != "0" ]] && debug_args+=(--disable-checkpoint)
[[ "$disable_transformer_ckpt" != "0" ]] && debug_args+=(--disable-transformer-ckpt)
[[ "$cuda_sync_debug" != "0" ]] && debug_args+=(--cuda-sync-debug)

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
MSDA_DISABLE_EXT="$msda_disable_ext" \
python3 -m method6.train \
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
  --det-loss-weight "$det_loss_weight" \
  --cls-loss-weight "$cls_loss_weight" \
  --bbox-loss-weight "$bbox_loss_weight" \
  --giou-loss-weight "$giou_loss_weight" \
  --tau-gt "$tau_gt" \
  --tau-aux "$tau_aux" \
  --role-loss-weight "$role_loss_weight" \
  --aux-loss-weight "$aux_loss_weight" \
  --role-gt-weight "$role_gt_weight" \
  --role-aux-weight "$role_aux_weight" \
  --role-none-weight "$role_none_weight" \
  --role-gate-mode "$role_gate_mode" \
  --stage1-det-weight "$stage1_det_weight" \
  --stage1-role-weight "$stage1_role_weight" \
  --stage1-aux-weight "$stage1_aux_weight" \
  --stage2-det-weight "$stage2_det_weight" \
  --stage2-role-weight "$stage2_role_weight" \
  --stage2-aux-weight "$stage2_aux_weight" \
  --stage2-warmup-ratio "$stage2_warmup_ratio" \
  "${stage1_pattern_args[@]}" \
  "${stage2_pattern_args[@]}" \
  --top-k "$top_k" \
  "${device_args[@]}" \
  "${hf_token_args[@]}" \
  "${debug_args[@]}" \
  "${extra_stage_args[@]}" \
  "$@"
