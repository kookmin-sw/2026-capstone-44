#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

ckpt_repo_id="Ideallll/AerialVG"
ckpt_filename="aerialvg.pth"
ckpt_dir="$aerialvg_model_dir"
ckpt_path="$aerialvg_checkpoint_path"
cuda_ready=0

ensure_cuda_ready() {
  [[ "$cuda_ready" == "1" ]] && return 0
  prepare_cuda_build "method6 setup" "setup_method6.sh"
  cuda_ready=1
}

groundingdino_ready() {
  python3 -c 'import torch, groundingdino._C'
}

method6_ops_ready() {
  python3 -c 'import sys, torch; from pathlib import Path; ops_dir = Path("method6/model/AerialVG/ops").resolve(); [sys.path.insert(0, str(p)) for p in (ops_dir, *sorted((ops_dir / "build").glob("lib.*"))) if str(p) not in sys.path]; import MultiScaleDeformableAttention'
}

checkpoint_ready() {
  [[ -s "$ckpt_path" ]] || return 1
  python3 - "$ckpt_path" <<'PY'
import sys

from capstone.checkpoint_utils import load_torch_checkpoint

load_torch_checkpoint(sys.argv[1], map_location="cpu")
PY
}

ensure_hf_hub() {
  python3 -c 'import huggingface_hub' >/dev/null 2>&1 || python3 -m pip install huggingface_hub
}

download_checkpoint() {
  mkdir -p "$ckpt_dir"
  echo "Downloading $ckpt_filename from $ckpt_repo_id to $ckpt_path"
  python3 -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["CKPT_REPO_ID"], repo_type="model", local_dir=os.environ["CKPT_DIR"], allow_patterns=[os.environ["CKPT_FILENAME"]], token=os.environ.get("HF_TOKEN") or None)'
  checkpoint_ready
}

dataset_ready() {
  local dataset_dir="$1"
  python3 capstone/prepare_aerialvg_local.py --local-dir "$dataset_dir" --verify-only
}

echo "Checking AerialVG dataset..."
if dataset_ready "$server_dataset_dir"; then
  local_dataset_dir="$server_dataset_dir"
  echo "AerialVG dataset is already available at $local_dataset_dir"
elif dataset_ready "$local_download_dir"; then
  local_dataset_dir="$local_download_dir"
  echo "AerialVG dataset is already available at $local_dataset_dir"
else
  echo "AerialVG dataset was not found in $server_dataset_dir"
  echo "AerialVG dataset was not found in $local_download_dir"
  echo "Running download_aerialvg.sh"
  bash ./download_aerialvg.sh
  source ./script_config.sh
  if dataset_ready "$server_dataset_dir"; then
    local_dataset_dir="$server_dataset_dir"
  elif dataset_ready "$local_download_dir"; then
    local_dataset_dir="$local_download_dir"
  else
    echo "AerialVG dataset is still unavailable after download." >&2
    exit 1
  fi
  echo "AerialVG dataset is ready at $local_dataset_dir"
fi

echo "Checking GroundingDINO CUDA extension..."
if groundingdino_ready; then
  echo "GroundingDINO CUDA extension is already available."
else
  ensure_cuda_ready
  echo "Building GroundingDINO CUDA extension with setup.py"
  python3 setup.py build_ext --inplace
  groundingdino_ready
  echo "GroundingDINO CUDA extension is ready."
fi

echo "Checking AerialVG pretrained checkpoint..."
if checkpoint_ready; then
  echo "AerialVG checkpoint already exists at $ckpt_path"
else
  if [[ -e "$ckpt_path" ]]; then
    echo "Existing checkpoint is invalid and will be re-downloaded: $ckpt_path"
    rm -f "$ckpt_path"
  fi
  ensure_hf_hub
  export CKPT_REPO_ID="$ckpt_repo_id"
  export CKPT_DIR="$ckpt_dir"
  export CKPT_FILENAME="$ckpt_filename"
  download_checkpoint
  echo "AerialVG checkpoint is ready at $ckpt_path"
fi

echo "Checking method6 CUDA op..."
if method6_ops_ready; then
  echo "method6 CUDA op is already available."
else
  ensure_cuda_ready
  echo "Building method6 CUDA op"
  bash ./build_method6_ops.sh
  method6_ops_ready
  echo "method6 CUDA op is ready."
fi

echo "setup_method6.sh completed successfully."
