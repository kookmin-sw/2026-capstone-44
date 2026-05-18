#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

repo_root="$(pwd)"
ops_dir="./method6/model/AerialVG/ops"

prepare_cuda_build "method6 custom op" "build_method6_ops.sh"

cd "$ops_dir"
python3 setup.py build_ext --inplace

cd "$repo_root"
python3 -c 'import sys, torch; from pathlib import Path; ops_dir = Path("method6/model/AerialVG/ops").resolve(); [sys.path.insert(0, str(p)) for p in (ops_dir, *sorted((ops_dir / "build").glob("lib.*"))) if str(p) not in sys.path]; import MultiScaleDeformableAttention; print("Loaded", MultiScaleDeformableAttention.__file__)'
