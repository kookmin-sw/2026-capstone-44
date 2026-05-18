#!/usr/bin/env bash
set -euo pipefail

target_user=sihaun
grace_seconds=3
list_only=false

[[ "${1:-}" == "--list" ]] && list_only=true

gpu_rows="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
app_rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true)"

if [[ -z "${app_rows// }" ]]; then
  echo "No GPU compute processes are running."
  exit 0
fi

declare -a pids=()
declare -a lines=()

while IFS=, read -r raw_uuid raw_pid raw_name raw_mem; do
  gpu_uuid="$(xargs <<<"$raw_uuid")"
  pid="$(xargs <<<"$raw_pid")"
  proc_name="$(xargs <<<"$raw_name")"
  used_mem="$(xargs <<<"$raw_mem")"

  [[ -n "$pid" ]] || continue
  [[ -d "/proc/$pid" ]] || continue

  owner="$(ps -o user:50= -p "$pid" | awk '{$1=$1; print}')"
  [[ "$owner" == "$target_user" ]] || continue

  gpu_info="$(awk -F', *' -v gpu_uuid="$gpu_uuid" '$2 == gpu_uuid {printf "gpu %s (%s)", $1, $3}' <<<"$gpu_rows")"
  [[ -n "$gpu_info" ]] || gpu_info="gpu ?"

  cmdline="$(ps -o cmd= -p "$pid" | awk '{$1=$1; print}')"

  pids+=("$pid")
  lines+=("$gpu_info | pid=$pid | mem=${used_mem} MiB | name=$proc_name | cmd=$cmdline")
done <<<"$app_rows"

if ((${#pids[@]} == 0)); then
  echo "No GPU compute processes owned by $target_user were found."
  exit 0
fi

echo "Found ${#pids[@]} GPU compute process(es) owned by $target_user:"
printf '%s\n' "${lines[@]}"

if [[ "$list_only" == true ]]; then
  exit 0
fi

echo "Sending SIGTERM to: ${pids[*]}"
kill "${pids[@]}" 2>/dev/null || true
sleep "$grace_seconds"

declare -a survivors=()
for pid in "${pids[@]}"; do
  [[ -d "/proc/$pid" ]] && survivors+=("$pid")
done

if ((${#survivors[@]} == 0)); then
  echo "All matching processes exited cleanly."
  exit 0
fi

echo "Still running after ${grace_seconds}s: ${survivors[*]}"
echo "Sending SIGKILL."
kill -9 "${survivors[@]}" 2>/dev/null || true

echo "Done."
