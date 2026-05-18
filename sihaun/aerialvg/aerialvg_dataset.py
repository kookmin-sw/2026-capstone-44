from __future__ import annotations

import sys
from pathlib import Path

import groundingdino.datasets.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capstone.aerialvg_hf import AerialVGHFDataset, DEFAULT_REPO_ID

TRAIN_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
EVAL_SCALES = [800]
MAX_SIZE = 1333


def normalize_split(split: str) -> str:
    lowered = split.lower()
    if lowered in {"validation", "valid", "dev"}:
        return "val"
    return lowered


def build_train_transforms():
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomResize(TRAIN_SCALES, max_size=MAX_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transforms():
    return T.Compose(
        [
            T.RandomResize(EVAL_SCALES, max_size=MAX_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_aerialvg_dataset(
    split: str,
    repo_id: str = DEFAULT_REPO_ID,
    hf_token: str | None = None,
    revision: str | None = None,
    annotation_file: str | None = None,
    image_root: str | None = None,
    transforms=None,
    max_samples: int | None = None,
):
    normalized_split = normalize_split(split)
    dataset_transforms = transforms
    if dataset_transforms is None:
        dataset_transforms = build_train_transforms() if normalized_split == "train" else build_eval_transforms()

    return AerialVGHFDataset(
        split=normalized_split,
        repo_id=repo_id,
        hf_token=hf_token,
        revision=revision,
        annotation_file=annotation_file,
        image_root=image_root,
        transforms=dataset_transforms,
        max_samples=max_samples,
    )
