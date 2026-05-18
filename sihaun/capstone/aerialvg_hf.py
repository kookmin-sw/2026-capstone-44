from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset

import groundingdino.datasets.transforms as T

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - handled at runtime with a clear error message.
    load_dataset = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - handled at runtime with a clear error message.
    hf_hub_download = None


DEFAULT_REPO_ID = "IPEC-COMMUNITY/AerialVG"
DEFAULT_SPLIT_FILES = {
    "train": "annotation/vg_train_odvg.jsonl",
    "val": "annotation/vg_val_odvg.jsonl",
    "test": "annotation/vg_test_odvg.jsonl",
}


def build_eval_transforms():
    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class AerialVGDatasetError(RuntimeError):
    """Raised when an AerialVG sample cannot be normalized into the expected format."""


class AerialVGHFDataset(Dataset):
    """AerialVG evaluation dataset loader.

    The loader is intentionally defensive because the gated Hugging Face dataset may
    be consumed in two different ways:
    1. Directly through `datasets.load_dataset(...)`, where records may already include
       an `image` feature.
    2. From a downloaded JSONL annotation file plus a separate image directory.

    In both cases this dataset returns a GroundingDINO-friendly `(image, target)` pair.
    """

    def __init__(
        self,
        split: str = "test",
        repo_id: str = DEFAULT_REPO_ID,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        annotation_file: Optional[str] = None,
        image_root: Optional[str] = None,
        transforms=None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split = split
        self.repo_id = repo_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.revision = revision
        self.annotation_file = annotation_file
        self.allow_hf_image_fallback = annotation_file is None
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None
        self.transforms = transforms or build_eval_transforms()

        self._records = self._load_records()
        if max_samples is not None:
            self._records = self._records[:max_samples]

    def _load_records(self):
        if self.annotation_file:
            annotation_path = Path(self.annotation_file).expanduser().resolve()
            if annotation_path.is_dir():
                split_file = DEFAULT_SPLIT_FILES.get(self.split)
                if split_file is None:
                    raise ValueError(
                        f"Unsupported split `{self.split}`. Expected one of: "
                        f"{', '.join(sorted(DEFAULT_SPLIT_FILES))}."
                    )
                relative_split = Path(split_file)
                candidates = [annotation_path / relative_split, annotation_path / relative_split.name]
                for candidate in candidates:
                    if candidate.exists():
                        annotation_path = candidate
                        break
                else:
                    raise FileNotFoundError(
                        f"Could not find split file for `{self.split}` under {annotation_path}."
                    )
            with annotation_path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle]

        if load_dataset is None:
            raise ImportError(
                "The `datasets` package is required to load AerialVG from Hugging Face. "
                "Install it in the current environment first."
            )

        split_file = DEFAULT_SPLIT_FILES.get(self.split)
        if split_file is None:
            raise ValueError(
                f"Unsupported split `{self.split}`. Expected one of: "
                f"{', '.join(sorted(DEFAULT_SPLIT_FILES))}."
            )

        try:
            dataset = load_dataset(
                self.repo_id,
                split=self.split,
                token=self.hf_token,
                revision=self.revision,
            )
            return [dataset[idx] for idx in range(len(dataset))]
        except Exception:
            dataset = load_dataset(
                self.repo_id,
                data_files={self.split: split_file},
                split=self.split,
                token=self.hf_token,
                revision=self.revision,
            )
            return [dataset[idx] for idx in range(len(dataset))]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        sample = self._records[index]
        image = self._extract_image(sample)
        query = self._extract_query(sample)
        gt_box = self._extract_gt_box(sample)
        image_id = self._extract_image_id(sample, index)

        width, height = image.size
        target = {
            "image_id": torch.as_tensor(image_id, dtype=torch.int64),
            "boxes": gt_box.unsqueeze(0),
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
            "caption": query,
            "query": query,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _extract_image(self, sample: Dict[str, Any]) -> Image.Image:
        image_value = sample.get("image")
        if image_value is not None:
            image = self._coerce_image(image_value)
            if image is not None:
                return image

        for key in ("filename", "image_path", "img_path", "path"):
            if key in sample:
                image = self._load_image_from_path(sample[key])
                if image is not None:
                    return image

        grounding = sample.get("grounding")
        if isinstance(grounding, dict):
            for key in ("image", "filename", "image_path", "img_path", "path"):
                if key in grounding:
                    image = self._coerce_image(grounding[key]) or self._load_image_from_path(
                        grounding[key]
                    )
                    if image is not None:
                        return image

        raise AerialVGDatasetError(
            "Could not resolve an image for the current AerialVG sample. "
            "Provide `--image-root` if the dataset record only contains filenames."
        )

    def _coerce_image(self, value: Any) -> Optional[Image.Image]:
        if isinstance(value, Image.Image):
            return value.convert("RGB")

        if isinstance(value, dict):
            image_bytes = value.get("bytes")
            if image_bytes is not None:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            path = value.get("path")
            if path:
                return self._load_image_from_path(path)

        if isinstance(value, str):
            return self._load_image_from_path(value)

        return None

    def _load_image_from_path(self, path_like: Any) -> Optional[Image.Image]:
        if not isinstance(path_like, str):
            return None

        candidate = Path(path_like)
        search_paths = [candidate]
        if self.image_root is not None:
            search_paths.append(self.image_root / candidate)

        for path in search_paths:
            if path.exists():
                return Image.open(path).convert("RGB")

        if not self.allow_hf_image_fallback:
            return None

        for remote_path in self._iter_remote_image_paths(candidate):
            image = self._load_image_from_hf(remote_path)
            if image is not None:
                return image
        return None

    def _iter_remote_image_paths(self, candidate: Path):
        seen = set()
        options = [candidate.as_posix(), candidate.name]
        if not candidate.as_posix().startswith("images/"):
            options.append(f"images/{candidate.name}")
            options.append(f"images/{candidate.as_posix().lstrip('./')}")

        for option in options:
            normalized = option.replace('\\', '/').lstrip('./')
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            yield normalized

    def _load_image_from_hf(self, remote_path: str) -> Optional[Image.Image]:
        if hf_hub_download is None:
            return None

        for local_only in (True, False):
            try:
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    token=self.hf_token,
                    revision=self.revision,
                    local_files_only=local_only,
                    etag_timeout=30,
                )
            except Exception:
                continue

            try:
                return Image.open(local_path).convert("RGB")
            except Exception:
                return None
        return None

    def _extract_query(self, sample: Dict[str, Any]) -> str:
        grounding = sample.get("grounding")
        if isinstance(grounding, dict):
            for key in ("caption", "query", "sentence", "text"):
                value = grounding.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        for key in ("caption", "query", "sentence", "text", "anno"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        raise AerialVGDatasetError("Could not find a grounding query/caption in the sample.")

    def _extract_gt_box(self, sample: Dict[str, Any]) -> torch.Tensor:
        regions = None
        grounding = sample.get("grounding")
        if isinstance(grounding, dict):
            regions = grounding.get("regions")
        if regions is None:
            regions = sample.get("regions")

        if not isinstance(regions, Iterable):
            raise AerialVGDatasetError("Could not find `regions` for the current sample.")

        for region in regions:
            if not isinstance(region, dict):
                continue
            for key in ("bbox", "box"):
                if key in region:
                    bbox = region[key]
                    if isinstance(bbox, Iterable):
                        bbox = list(bbox)
                        if len(bbox) == 4:
                            return torch.as_tensor(bbox, dtype=torch.float32)

        raise AerialVGDatasetError("Could not find a valid 4D bounding box in sample regions.")

    def _extract_image_id(self, sample: Dict[str, Any], index: int) -> int:
        for key in ("image_id", "id", "sample_id"):
            value = sample.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return index
