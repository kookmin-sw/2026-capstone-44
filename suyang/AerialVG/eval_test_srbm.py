"""
Quick eval: AerialVG backbone + our SRBM, on TEST set.
Reuses train_srbm_only.py infrastructure but only runs evaluate().
"""
import sys, os, json, argparse, torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

# Import everything from train_srbm_only
from train_srbm_only import (
    AerialVGHFDataset, collate_fn, evaluate,
    TRANSFORM_VAL,
)
from model import build_model
from util.slconfig import SLConfig
from util.misc import clean_state_dict
import argparse as ap

CONFIG = "./config/config_cfg.py"
CKPT   = "./output/srbm_only/best.pth"
ANNO   = "/data2/huggingface/AerialVG/annotation/vg_test_odvg.jsonl"
IMG_ROOT = "/data2/huggingface/AerialVG/images"
DEVICE = "cuda:0"

os.chdir(Path(__file__).parent)

cfg = SLConfig.fromfile(CONFIG)
cfg_dict = cfg._cfg_dict.to_dict()
model_args = ap.Namespace(**cfg_dict)
model_args.use_srbm    = True
model_args.srbm_layers = 3
model_args.topk        = 15

print(f"Building model with SRBM...")
model = build_model(model_args).to(DEVICE)

print(f"Loading: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = clean_state_dict(ckpt.get("model", ckpt))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")

ds = AerialVGHFDataset(ANNO, IMG_ROOT, transform=TRANSFORM_VAL)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)
print(f"Test samples: {len(ds)}")

top1, top5 = evaluate(model, loader, DEVICE, topk=5)
print(f"\n{'='*55}")
print(f"  AerialVG + our SRBM (TEST set)")
print(f"  Top-1: {top1*100:.2f}%")
print(f"  Top-5: {top5*100:.2f}%")
print(f"{'='*55}")
