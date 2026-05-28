import os
from typing import Optional, Tuple

import PIL.Image
import torch
from torch.utils.data import Dataset

import groundingdino.datasets.transforms as T
from groundingdino.util.vl_utils import create_positive_map_from_span


def _find_phrase_span(caption: str, phrase: str) -> Tuple[int, int]:
    """Return (start, end) char indices of phrase in the original caption string.
    Searches case-insensitively; returned indices reference the original caption.
    """
    cl = caption.lower()
    pl = phrase.lower()

    # 1. Exact case-insensitive match
    idx = cl.find(pl)
    if idx != -1:
        return idx, idx + len(phrase)

    # 2. Strip leading article
    for art in ("a ", "an ", "the "):
        if pl.startswith(art):
            stripped = pl[len(art):]
            idx = cl.find(stripped)
            if idx != -1:
                return idx, idx + len(stripped)

    # 3. First word match
    first_word = pl.split()[0] if pl.split() else pl
    idx = cl.find(first_word)
    if idx != -1:
        return idx, idx + len(phrase)

    # 4. Fallback: start of caption (ensure end > start)
    end = min(len(phrase), len(caption))
    if end <= 0:
        end = min(1, len(caption))
    return 0, end


class AerialVGDataset(Dataset):
    """PyTorch Dataset wrapping a HuggingFace Dataset split for AerialVG.

    Args:
        hf_split: A HuggingFace Dataset split object
                  (e.g. load_dataset('/data2/huggingface/AerialVG')['train']).
        image_dir: Path to the directory containing image files.
        tokenizer: A fast HuggingFace tokenizer (must support char_to_token()).
        transforms: Composed transforms from groundingdino.datasets.transforms.
        max_text_len: Maximum text token length (must match model config).
    """

    def __init__(
        self,
        hf_split,
        image_dir: str,
        tokenizer,
        transforms=None,
        max_text_len: int = 256,
    ):
        self.data = hf_split
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        filename = item["filename"]
        h = item["height"]
        w = item["width"]
        grounding = item["grounding"]
        caption: str = grounding["caption"]
        regions = grounding["regions"]

        # --- Load image ---
        img_path = os.path.join(self.image_dir, filename)
        image = PIL.Image.open(img_path).convert("RGB")

        # --- Build boxes [N, 4] in xyxy pixel coords ---
        if len(regions) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            token_spans = []
        else:
            boxes = torch.tensor(
                [r["bbox"] for r in regions], dtype=torch.float32
            )  # [N, 4] xyxy
            # Clamp to image boundary
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            # --- Find char-level spans for each phrase in the caption ---
            token_spans = []
            for region in regions:
                phrase = region["phrase"]
                start, end = _find_phrase_span(caption, phrase)
                token_spans.append([(start, end)])

        # --- Build positive_map [N, max_text_len] ---
        # Tokenize with fast tokenizer so char_to_token() is available
        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_text_len,
        )
        positive_map = create_positive_map_from_span(
            tokenized, token_spans, max_text_len=self.max_text_len
        )  # [N, max_text_len], row-normalized

        # --- Build target dict ---
        target = {
            "boxes": boxes,                                          # [N, 4] xyxy pixels
            "positive_map": positive_map,                            # [N, max_text_len]
            "caption": caption,                                      # str
            "labels": torch.zeros(len(regions), dtype=torch.long),  # [N]
            "orig_size": torch.tensor([h, w]),                       # [2]
            "image_id": torch.tensor([idx]),                         # [1]
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def build_train_transforms() -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomSelect(
            T.RandomResize(
                [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                max_size=1333,
            ),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600, respect_boxes=False),
                T.RandomResize(
                    [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                    max_size=1333,
                ),
            ]),
        ),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
