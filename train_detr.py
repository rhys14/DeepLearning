"""Train GalaxyDETR (run from the repo root).

    python train_detr.py --config configs/detr_config.yaml
    python train_detr.py --config configs/detr_config.yaml training.epochs=20 model.backbone=mobilenetv4_conv_medium
"""
import argparse
import os

import torch

from datasets.galaxy10_detr_dataset import CLASS_NAMES, build_dataloaders
from Networks.galaxydetr import build_model
from training.detr_loss import build_criterion
from training.detr_validater import evaluate_model
from training.detr_trainer import build_optimizer, train_model
from utils.config import apply_overrides, load_config
from utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/detr_config.yaml')
    ap.add_argument('overrides', nargs='*', help='key=value pairs, e.g. training.epochs=20')
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)

    torch.manual_seed(cfg['seed'])
    device = get_device()
    train_loader, val_loader = build_dataloaders(cfg)

    num_classes = len(CLASS_NAMES)
    model = build_model(cfg, num_classes).to(device)
    print(f'Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    criterion = build_criterion(cfg, num_classes).to(device)
    optimizer = build_optimizer(model, cfg)

    os.makedirs(os.path.dirname(cfg['checkpoint']) or '.', exist_ok=True)
    train_model(model, criterion, optimizer, train_loader, val_loader, device,
                cfg['training']['epochs'], cfg['checkpoint'], cfg['training']['grad_clip'])
    evaluate_model(model, val_loader, device, cfg['checkpoint'], CLASS_NAMES)


if __name__ == '__main__':
    main()
