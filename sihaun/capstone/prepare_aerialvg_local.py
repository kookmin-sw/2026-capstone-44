from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "IPEC-COMMUNITY/AerialVG"
DEFAULT_LOCAL_DIR = Path("/data2/huggingface/AerialVG")
SPLIT_FILES = (
    "annotation/vg_train_odvg.jsonl",
    "annotation/vg_val_odvg.jsonl",
    "annotation/vg_test_odvg.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download and verify a local AerialVG dataset mirror")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=os.getenv("LOCAL_DATASET_DIR", str(DEFAULT_LOCAL_DIR)))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-sleep", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--etag-timeout", type=int, default=30)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def verify_local_dataset(local_dir: Path) -> tuple[bool, dict]:
    ann_root = local_dir / "annotation"
    img_root = local_dir / "images"
    missing_annotations: list[str] = []
    required_images: set[str] = set()

    for split_file in SPLIT_FILES:
        ann_path = local_dir / split_file
        if not ann_path.exists():
            missing_annotations.append(split_file)
            continue
        with ann_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                filename = record.get("filename") or record.get("grounding", {}).get("filename")
                if filename:
                    required_images.add(filename)

    missing_images = sorted(
        filename for filename in required_images if not (img_root / filename).exists()
    )
    stats = {
        "local_dir": str(local_dir.resolve()),
        "cached_images": sum(1 for _ in img_root.glob("*.jpg")),
        "required_unique_images": len(required_images),
        "missing_annotations": missing_annotations,
        "missing_unique_images": len(missing_images),
        "sample_missing_images": missing_images[:10],
    }
    ok = not missing_annotations and not missing_images
    return ok, stats


def print_stats(stats: dict) -> None:
    print(f"local_dir={stats['local_dir']}")
    print(f"cached_images={stats['cached_images']}")
    print(f"required_unique_images={stats['required_unique_images']}")
    print(f"missing_annotations={len(stats['missing_annotations'])}")
    if stats["missing_annotations"]:
        print("sample_missing_annotations=", stats["missing_annotations"][:5])
    print(f"missing_unique_images={stats['missing_unique_images']}")
    if stats["sample_missing_images"]:
        print("sample_missing_images=", stats["sample_missing_images"])


def download_once(args: argparse.Namespace, local_dir: Path) -> None:
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=args.hf_token,
        allow_patterns=["annotation/*", "images/*"],
        etag_timeout=args.etag_timeout,
        max_workers=args.max_workers,
    )


def main() -> int:
    args = parse_args()
    args.hf_token = args.hf_token or None
    local_dir = Path(args.local_dir).expanduser().resolve()

    ok, stats = verify_local_dataset(local_dir)
    print_stats(stats)
    if ok:
        print("AerialVG local dataset is complete.")
        return 0
    if args.verify_only:
        print("AerialVG local dataset is incomplete.", file=sys.stderr)
        return 1

    local_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.retries + 1):
        print(f"Download attempt {attempt}/{args.retries} with max_workers={args.max_workers}")
        try:
            download_once(args, local_dir)
        except Exception as exc:
            print(f"download_error={exc}", file=sys.stderr)

        ok, stats = verify_local_dataset(local_dir)
        print_stats(stats)
        if ok:
            print("AerialVG local dataset is complete.")
            return 0
        if attempt < args.retries:
            print(f"Dataset still incomplete. Sleeping {args.retry_sleep}s before retry.")
            time.sleep(args.retry_sleep)

    print("AerialVG local dataset is still incomplete after all retries.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
