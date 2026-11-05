"""Evaluate a trained GalaxyDETR checkpoint on the validation split (run from the repo root).

    python evaluate_detr.py --config configs/detr_config.yaml
    python evaluate_detr.py --config configs/detr_config.yaml checkpoint=models/other_run.pth
    python evaluate_detr.py --split val        # default: test (falls back to val if there is no test set)

Use the same data.val_fraction / data.test_fraction / seed as the training run, or the splits won't match.
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
    ap.add_argument('--split', choices=['test', 'val', 'both'], default='test')
    ap.add_argument('overrides', nargs='*', help='key=value pairs, e.g. checkpoint=models/run2.pth')
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg['model']['pretrained'] = False  # weights come from the checkpoint; skip the ImageNet download

    torch.manual_seed(cfg['seed'])
    device = get_device()
    _, val_loader, test_loader = build_dataloaders(cfg)  # same seed + fractions -> same split as training
    model = build_model(cfg, len(CLASS_NAMES)).to(device)
    split = args.split
    if test_loader is None and split != 'val':
        print('No test set (data.test_fraction is 0); evaluating the validation set.')
        split = 'val'
    if split in ('val', 'both'):
        evaluate_model(model, val_loader, device, cfg['checkpoint'], CLASS_NAMES, 'Validation')
    if split in ('test', 'both'):
        evaluate_model(model, test_loader, device, cfg['checkpoint'], CLASS_NAMES, 'Test')


if __name__ == '__main__':
    main()
