#!/usr/bin/env bash

require_local_dataset() {
  python3 capstone/prepare_aerialvg_local.py --local-dir "$local_dataset_dir" --verify-only \
    || {
      echo "Local AerialVG dataset is incomplete. Run: bash download_aerialvg.sh" >&2
      exit 1
    }
}

pick_checkpoint_path() {
  local preferred_path="$1"
  local fallback_path="$2"
  [[ -f "$preferred_path" ]] && printf '%s\n' "$preferred_path" || printf '%s\n' "$fallback_path"
}

find_nvcc_bin() {
  local candidate

  candidate="$(command -v nvcc || true)"
  [[ -n "$candidate" ]] && printf '%s\n' "$candidate" && return 0

  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$candidate" ]] && printf '%s\n' "$candidate" && return 0
  done

  return 1
}

ensure_ninja() {
  python3 -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("ninja") else 1)' \
    || python3 -m pip install ninja
}

ensure_visible_cuda() {
  local script_name="$1"

  python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || {
      echo "torch.cuda.is_available() is false in the current environment." >&2
      echo "Choose a usable GPU first, for example: CUDA_VISIBLE_DEVICES=0 bash $script_name" >&2
      echo "If you leave CUDA_VISIBLE_DEVICES unset, the runtime will use the default visible cuda:0 when one exists." >&2
      echo "If you really need to compile without a visible GPU, set FORCE_CUDA=1 explicitly yourself." >&2
      exit 1
    }
}

prepare_cuda_build() {
  local target_name="$1"
  local script_name="$2"
  local nvcc_bin

  nvcc_bin="$(find_nvcc_bin)" \
    || {
      echo "nvcc was not found in PATH or common CUDA install locations." >&2
      echo "Install a CUDA toolkit and set CUDA_HOME before rebuilding the $target_name." >&2
      exit 1
    }

  export PATH="$(dirname "$nvcc_bin"):$PATH"
  export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$nvcc_bin")/.." && pwd)}"

  echo "Using nvcc: $nvcc_bin"
  echo "Using CUDA_HOME: $CUDA_HOME"

  ensure_ninja
  ensure_visible_cuda "$script_name"
}
