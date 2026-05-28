import torch
from torch.utils.data import Dataset


class CachedTopKDataset(Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path, map_location="cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        output = {
            "topk_logits": item["topk_logits"].float(),
            "topk_boxes": item["topk_boxes"].float(),
            "topk_scores": item["topk_scores"].float(),
            "gt_box": item["gt_box"].float(),
            "positive_idx": item["positive_idx"].long(),
        }

        for key in ["image_path", "img_path", "image", "file_name", "filename"]:
            if key in item:
                output["image_path"] = item[key]
                break

        if "caption" in item:
            output["caption"] = item["caption"]

        return output


def cached_collate_fn(batch):
    output = {
        "topk_logits": torch.stack([b["topk_logits"] for b in batch], dim=0),
        "topk_boxes": torch.stack([b["topk_boxes"] for b in batch], dim=0),
        "topk_scores": torch.stack([b["topk_scores"] for b in batch], dim=0),
        "gt_box": torch.stack([b["gt_box"] for b in batch], dim=0),
        "positive_idx": torch.stack([b["positive_idx"] for b in batch], dim=0),

        
        "gt_boxes": torch.stack([b["gt_box"] for b in batch], dim=0),
        "labels": torch.stack([b["positive_idx"] for b in batch], dim=0),
    }

    if any("image_path" in b for b in batch):
        output["image_path"] = [b.get("image_path", "") for b in batch]

    if any("caption" in b for b in batch):
        output["caption"] = [b.get("caption", "") for b in batch]

    return output