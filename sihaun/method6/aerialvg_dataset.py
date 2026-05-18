from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable

import groundingdino.datasets.transforms as T
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capstone.aerialvg_hf import AerialVGDatasetError, AerialVGHFDataset, DEFAULT_REPO_ID

TRAIN_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
EVAL_SCALES = [800]
MAX_SIZE = 1333


class Method2AerialVGHFDataset(AerialVGHFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original_count = len(self._records)
        self._records = [sample for sample in self._records if self._has_valid_query(sample)]
        filtered_count = original_count - len(self._records)
        if filtered_count > 0:
            warnings.warn(
                f"Filtered {filtered_count} AerialVG sample(s) without a valid query/caption from split `{self.split}`.",
                UserWarning,
            )

    def _has_valid_query(self, sample: Dict) -> bool:
        grounding = sample.get("grounding")
        if isinstance(grounding, dict):
            for key in ("caption", "query", "sentence", "text"):
                value = grounding.get(key)
                if isinstance(value, str) and value.strip():
                    return True

        for key in ("caption", "query", "sentence", "text", "anno"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False

    def __getitem__(self, index: int):
        sample = self._records[index]
        image = self._extract_image(sample)
        query = self._extract_query(sample)
        regions = self._extract_regions(sample)
        boxes = self._extract_boxes_from_regions(regions)
        image_id = self._extract_image_id(sample, index)
        gt_phrase, aux_phrases = self._extract_region_phrases(regions)
        aux_relations = self._extract_aux_relations(regions)

        width, height = image.size
        target = {
            "image_id": torch.as_tensor(image_id, dtype=torch.int64),
            "boxes": boxes,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "caption": query,
            "query": query,
            "gt_phrase": gt_phrase,
            "aux_phrases": aux_phrases,
            "aux_relations": aux_relations,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        transformed_boxes = target["boxes"]
        if transformed_boxes.ndim != 2 or transformed_boxes.shape[0] == 0:
            raise AerialVGDatasetError(
                "Expected transformed target boxes to have shape [num_regions, 4]."
            )

        target["gt_box"] = transformed_boxes[0].clone()
        target["aux_boxes"] = transformed_boxes[1:].clone()
        target["num_aux"] = torch.as_tensor(
            int(max(transformed_boxes.shape[0] - 1, 0)), dtype=torch.int64
        )
        return image, target

    def _extract_regions(self, sample: Dict):
        regions = None
        grounding = sample.get("grounding")
        if isinstance(grounding, dict):
            regions = grounding.get("regions")
        if regions is None:
            regions = sample.get("regions")

        if not isinstance(regions, Iterable):
            raise AerialVGDatasetError("Could not find `regions` for the current sample.")

        normalized_regions = [region for region in regions if isinstance(region, dict)]
        if not normalized_regions:
            raise AerialVGDatasetError(
                "Could not find valid region dictionaries in the current sample."
            )
        return normalized_regions

    def _extract_boxes_from_regions(self, regions) -> torch.Tensor:
        boxes = []
        for region in regions:
            for key in ("bbox", "box"):
                if key not in region:
                    continue
                bbox = region[key]
                if not isinstance(bbox, Iterable):
                    continue
                bbox = list(bbox)
                if len(bbox) == 4:
                    boxes.append(bbox)
                    break

        if not boxes:
            raise AerialVGDatasetError(
                "Could not find a valid 4D bounding box in sample regions."
            )
        return torch.as_tensor(boxes, dtype=torch.float32)

    def _extract_region_phrases(self, regions):
        phrases = []
        for region in regions:
            phrase = region.get("phrase")
            phrases.append(phrase.strip() if isinstance(phrase, str) else "")

        gt_phrase = phrases[0] if phrases else ""
        aux_phrases = phrases[1:] if len(phrases) > 1 else []
        return gt_phrase, aux_phrases

    def _extract_aux_relations(self, regions):
        relations = []
        for region in regions[1:]:
            value = region.get("realation", region.get("relation", ""))
            relations.append(value.strip() if isinstance(value, str) else "")
        return relations


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

    return Method2AerialVGHFDataset(
        split=normalized_split,
        repo_id=repo_id,
        hf_token=hf_token,
        revision=revision,
        annotation_file=annotation_file,
        image_root=image_root,
        transforms=dataset_transforms,
        max_samples=max_samples,
    )
