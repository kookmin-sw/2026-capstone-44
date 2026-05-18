#!/usr/bin/env bash
set -euo pipefail

source ./script_config.sh
source ./script_helpers.sh

repo_root="$(pwd)"
ops_dir="./aerialvg/model/AerialVG/ops"

prepare_cuda_build "aerialvg custom op" "build_aerialvg_ops.sh"

cd "$ops_dir"
python3 setup.py build_ext --inplace

cd "$repo_root"
python3 -c 'import sys, torch; from pathlib import Path; ops_dir = Path("aerialvg/model/AerialVG/ops").resolve(); [sys.path.insert(0, str(p)) for p in (ops_dir, *sorted((ops_dir / "build").glob("lib.*"))) if str(p) not in sys.path]; import MultiScaleDeformableAttention; print("Loaded", MultiScaleDeformableAttention.__file__)'
