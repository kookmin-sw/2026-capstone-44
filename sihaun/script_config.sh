#!/usr/bin/env bash

if [[ -f .env ]]; then
  set -a
  source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env || true)
  set +a
fi

hf_token="${hf_token:-${HF_TOKEN:-}}"
dataset_repo="${dataset_repo:-${DATASET_REPO:-IPEC-COMMUNITY/AerialVG}}"
# Leave unset to let each Python entrypoint auto-pick `cuda`/`cuda:0` when available,
# otherwise `cpu`. Set `device`/`DEVICE` explicitly to override.
device="${device-${DEVICE-}}"
msda_disable_ext="${msda_disable_ext:-${MSDA_DISABLE_EXT:-1}}"
server_dataset_dir=/data2/huggingface/AerialVG
local_download_dir=./data/AerialVG
model_storage_dir="${model_storage_dir:-${MODEL_STORAGE_DIR:-./checkpoints}}"
aerialvg_model_dir="$model_storage_dir/aerialvg"
method4_model_dir="$model_storage_dir/method4"
method6_model_dir="$model_storage_dir/method6"
aerialvg_outputs_dir="$aerialvg_model_dir/outputs"
method4_outputs_dir="$method4_model_dir/outputs"
method6_outputs_dir="$method6_model_dir/outputs"
aerialvg_checkpoint_path="$aerialvg_model_dir/aerialvg.pth"
aerialvg_output_dir="$aerialvg_outputs_dir/run"
method4_output_dir="$method4_outputs_dir/run"
method6_output_dir="$method6_outputs_dir/run"

# GPU visibility: leave unset to keep the default visible GPU order, so `cuda`
# resolves to the first visible device. Set one of these variables to pin a GPU.
legacy_cuda_visible_devices="${CUDA_VISIABLE_DEVICES:-}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -n "$legacy_cuda_visible_devices" && "${CUDA_VISIBLE_DEVICES}" != "$legacy_cuda_visible_devices" ]]; then
  echo "Warning: ignoring CUDA_VISIABLE_DEVICES=$legacy_cuda_visible_devices because CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} is also set." >&2
fi
cuda_visible_devices="${cuda_visible_devices-${CUDA_VISIBLE_DEVICES-${legacy_cuda_visible_devices-}}}"

# dataset location: leave only one of the next two lines uncommented.
local_dataset_dir="${local_dataset_dir:-${LOCAL_DATASET_DIR:-$server_dataset_dir}}"  # server
# local_dataset_dir="${local_dataset_dir:-${LOCAL_DATASET_DIR:-$local_download_dir}}" # home

if [[ "$local_dataset_dir" == "$server_dataset_dir" && ! -d "$server_dataset_dir/annotation" && -d "$local_download_dir/annotation" ]]; then
  local_dataset_dir="$local_download_dir"
fi

[[ -n "$cuda_visible_devices" ]] && export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
if [[ -n "$hf_token" ]]; then
  export HF_TOKEN="$hf_token"
else
  unset HF_TOKEN
fi
