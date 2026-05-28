import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import groundingdino.datasets.transforms as T


class AerialVGJsonlDataset(Dataset):
    def __init__(self, jsonl_path, image_root):
        self.image_root = image_root
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                filename = item["filename"]
                height = item["height"]
                width = item["width"]

                grounding = item["grounding"]
                caption = grounding["caption"]
                regions = grounding["regions"]

                for region in regions:
                    self.samples.append({
                        "filename": filename,
                        "height": height,
                        "width": width,
                        "caption": caption,
                        "bbox": region["bbox"],
                        "phrase": region.get("phrase", ""),
                        "relation": region.get("relation", None),
                    })

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = os.path.join(self.image_root, item["filename"])
        image_pil = Image.open(image_path).convert("RGB")

        w, h = image_pil.size

        image_tensor, _ = self.transform(image_pil, None)

        bbox = torch.tensor(item["bbox"], dtype=torch.float32)

        x1, y1, x2, y2 = bbox

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        gt_box = torch.tensor(
            [cx, cy, bw, bh],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "caption": item["caption"],
            "phrase": item["phrase"],
            "relation": item["relation"],
            "gt_box": gt_box,
            "filename": item["filename"],
        }


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "phrases": [b["phrase"] for b in batch],
        "relations": [b["relation"] for b in batch],
        "gt_boxes": torch.stack([b["gt_box"] for b in batch], dim=0),
        "filenames": [b["filename"] for b in batch],
    }