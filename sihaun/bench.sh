#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

bash ./train_method6.sh "$@"
bash ./infer_method6.sh
