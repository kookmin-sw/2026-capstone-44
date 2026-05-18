#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh

max_workers=8
retries=20
retry_sleep=15
etag_timeout=30

if python3 capstone/prepare_aerialvg_local.py --local-dir "$server_dataset_dir" --verify-only; then
  echo "Using existing server dataset at $server_dataset_dir"
  exit 0
fi

echo "Server dataset not found at $server_dataset_dir"
echo "Downloading local copy to $local_download_dir"

cmd=(
  python3 capstone/prepare_aerialvg_local.py
  --local-dir "$local_download_dir"
  --max-workers "$max_workers"
  --retries "$retries"
  --retry-sleep "$retry_sleep"
  --etag-timeout "$etag_timeout"
)

[[ -n "$hf_token" ]] && cmd+=(--hf-token "$hf_token")
cmd+=("$@")

"${cmd[@]}"
