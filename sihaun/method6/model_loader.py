from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from capstone.checkpoint_utils import load_torch_checkpoint, normalize_aerialvg_state_dict_keys
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD6_ROOT = Path(__file__).resolve().parent
for path in (METHOD6_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from .model import build_model

DEFAULT_CONFIG_FILE = METHOD6_ROOT / "config" / "default.py"
LOCAL_BERT_CACHE = REPO_ROOT / ".hf-cache" / "bert-base-uncased"
DEFAULT_TRAINABLE_PATTERNS = (
    "role_evidence_module",
)


def resolve_repo_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None

    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [Path.cwd(), REPO_ROOT, METHOD6_ROOT]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (REPO_ROOT / candidate).resolve()


def load_method6_config(
    config_file: str | Path | None = None,
    *,
    use_checkpoint: bool | None = None,
    use_transformer_ckpt: bool | None = None,
    role_gate_mode: str | None = None,
):
    resolved = resolve_repo_path(config_file or DEFAULT_CONFIG_FILE)
    cfg = SLConfig.fromfile(str(resolved))
    if getattr(cfg, "text_encoder_type", None) == "bert-base-uncased" and LOCAL_BERT_CACHE.exists():
        cfg.text_encoder_type = str(LOCAL_BERT_CACHE)
    if use_checkpoint is not None:
        cfg.use_checkpoint = bool(use_checkpoint)
    if use_transformer_ckpt is not None:
        cfg.use_transformer_ckpt = bool(use_transformer_ckpt)
    if role_gate_mode is not None:
        cfg.role_gate_mode = role_gate_mode
    return cfg


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def promote_tensor_attributes_to_parameters(model: nn.Module, attribute_names: Iterable[str] = ("sim_param",)):
    promoted = []
    for name in attribute_names:
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
            setattr(model, name, nn.Parameter(value.detach().clone().float()))
            promoted.append(name)
    return promoted


def build_method6_model(
    config_file: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    strict: bool = False,
    use_checkpoint: bool | None = None,
    use_transformer_ckpt: bool | None = None,
    role_gate_mode: str | None = None,
):
    cfg = load_method6_config(
        config_file,
        use_checkpoint=use_checkpoint,
        use_transformer_ckpt=use_transformer_ckpt,
        role_gate_mode=role_gate_mode,
    )
    cfg.device = device
    model = build_model(cfg)
    promoted = promote_tensor_attributes_to_parameters(model)

    load_info = {
        "checkpoint_path": None,
        "missing_keys": [],
        "unexpected_keys": [],
        "promoted_attributes": promoted,
    }

    resolved_checkpoint = resolve_repo_path(checkpoint_path)
    if resolved_checkpoint is not None:
        checkpoint = load_torch_checkpoint(resolved_checkpoint, map_location="cpu")
        state_dict = clean_state_dict(_extract_state_dict(checkpoint))
        state_dict = normalize_aerialvg_state_dict_keys(state_dict, model.state_dict().keys())
        load_result = model.load_state_dict(state_dict, strict=strict)
        load_info["checkpoint_path"] = str(resolved_checkpoint)
        load_info["missing_keys"] = list(load_result.missing_keys)
        load_info["unexpected_keys"] = list(load_result.unexpected_keys)

    return model.to(device), cfg, load_info

def freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_by_name(model: nn.Module, patterns: Sequence[str]):
    matched = []
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad_(True)
            matched.append(name)
    return matched


def configure_trainable_parameters(model: nn.Module, trainable_patterns: Sequence[str] | None = None):
    patterns = tuple(trainable_patterns or DEFAULT_TRAINABLE_PATTERNS)
    freeze_all_parameters(model)
    matched = unfreeze_by_name(model, patterns)
    return patterns, matched


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
    return sum(parameter.numel() for parameter in parameters)
