from __future__ import annotations

from pathlib import Path

import torch


def _format_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def describe_checkpoint(path_like: str | Path) -> str:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        return str(path)
    return f"{path} ({_format_size(path.stat().st_size)})"


def validate_torch_checkpoint(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file was not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint file is empty: {path}")

    return path


def load_torch_checkpoint(path_like: str | Path, *, map_location: str = "cpu"):
    path = validate_torch_checkpoint(path_like)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except RuntimeError as exc:
        message = str(exc)
        if "PytorchStreamReader failed" in message or "failed finding central directory" in message:
            raise RuntimeError(
                f"Failed to load checkpoint {describe_checkpoint(path)}. "
                "The file is very likely incomplete or corrupted. "
                "Delete it and re-download it, then retry."
            ) from exc
        raise RuntimeError(f"Failed to load checkpoint {describe_checkpoint(path)}: {exc}") from exc


def normalize_aerialvg_state_dict_keys(state_dict, model_state_keys):
    model_key_set = set(model_state_keys)
    normalized = {}

    for key, value in state_dict.items():
        normalized_key = key
        if key not in model_key_set and key.startswith("bert."):
            legacy_suffix = key[len("bert.") :]
            candidate = f"bert.bert_model.{legacy_suffix}"
            if candidate in model_key_set:
                normalized_key = candidate
        normalized[normalized_key] = value

    return normalized
