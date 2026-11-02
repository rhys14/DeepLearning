"""Evaluate a trained GalaxyDETR checkpoint on the validation split (run from the repo root).

    python evaluate_detr.py --config configs/detr_config.yaml
    python evaluate_detr.py --config configs/detr_config.yaml checkpoint=models/other_run.pth
"""
import argparse

import torch

from datasets.galaxy10_detr_dataset import CLASS_NAMES, build_dataloaders
from Networks.galaxydetr import build_model
from training.detr_validater import evaluate_model
from utils.config import apply_overrides, load_config
from utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/detr_config.yaml')
    ap.add_argument('overrides', nargs='*', help='key=value pairs, e.g. checkpoint=models/run2.pth')
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg['model']['pretrained'] = False  # weights come from the checkpoint; skip the ImageNet download

    torch.manual_seed(cfg['seed'])
    device = get_device()
    _, val_loader = build_dataloaders(cfg)  # same seed -> same split as training
    model = build_model(cfg, len(CLASS_NAMES)).to(device)
    evaluate_model(model, val_loader, device, cfg['checkpoint'], CLASS_NAMES)


if __name__ == '__main__':
    main()
