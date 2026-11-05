"""Train GalaxyDETR (run from the repo root).

    python train_detr.py --config configs/detr_config.yaml
    python train_detr.py --config configs/detr_config.yaml training.epochs=20 model.backbone=mobilenetv4_conv_medium

Each run is named galaxy_detr_<YYYYmmdd_HHMMSS> and saves:
    models/<run_name>.pth          best weights + config + class names (load with Networks.galaxydetr.load_galaxydetr)
    models/galaxy_detr.pth         copy of the above, so evaluate_detr.py / predict_detr.py pick up the latest run
    results/<run_name>/report.md   results write-up (+ metrics.json, history.csv, loss_curve.png, config.yaml)
    results/<run_name>/history.json  per-epoch losses, updated every epoch (used by make_report_detr.py)
"""
import argparse
import copy
import os
import shutil
from datetime import datetime

import torch

from datasets.galaxy10_detr_dataset import CLASS_NAMES, build_dataloaders
from Networks.galaxydetr import build_model
from training.detr_loss import build_criterion
from training.detr_validater import evaluate_model
from training.detr_trainer import build_optimizer, train_model
from utils.checkpoint import save_checkpoint
from utils.config import apply_overrides, load_config
from utils.detr_report import write_report
from utils.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/detr_config.yaml')
    ap.add_argument('overrides', nargs='*', help='key=value pairs, e.g. training.epochs=20')
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)

    started = datetime.now()
    run_name = f"galaxy_detr_{started:%Y%m%d_%H%M%S}"
    out = cfg.get('output', {})
    models_dir, results_dir = out.get('models_dir', 'models'), out.get('results_dir', 'results')
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, f'{run_name}.pth')
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print('Run:', run_name)

    torch.manual_seed(cfg['seed'])
    device = get_device()
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    num_classes = len(CLASS_NAMES)
    model = build_model(cfg, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {n_params}')
    criterion = build_criterion(cfg, num_classes).to(device)
    optimizer = build_optimizer(model, cfg)

    saved_cfg = copy.deepcopy(cfg)  # stored inside the checkpoint so the model can be rebuilt anywhere

    def save_best(model, epoch, val_loss):
        save_checkpoint(weights_path, model, saved_cfg, CLASS_NAMES,
                        run_name=run_name, best_epoch=epoch, best_val_loss=float(val_loss))

    history = train_model(model, criterion, optimizer, train_loader, val_loader, device,
                          cfg['training']['epochs'], save_best, cfg['training']['grad_clip'],
                          cfg['training'].get('log_every', 50), os.path.join(run_dir, 'history.json'))
    metrics = evaluate_model(model, val_loader, device, weights_path, CLASS_NAMES, 'Validation')
    test_metrics = None
    if test_loader is not None:  # scored once, with the checkpoint validation picked
        test_metrics = evaluate_model(model, test_loader, device, weights_path, CLASS_NAMES, 'Test')

    os.makedirs(os.path.dirname(cfg['checkpoint']) or '.', exist_ok=True)
    shutil.copyfile(weights_path, cfg['checkpoint'])

    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else device.type
    setup = {'started': f'{started:%Y-%m-%d %H:%M}', 'device': device_name, 'n_params': n_params,
             'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
             'n_test': len(test_loader.dataset) if test_loader is not None else 0,
             'weights_path': weights_path.replace(os.sep, '/')}
    print(f'Saved weights: {weights_path} (also copied to {cfg["checkpoint"]})')
    try:
        report = write_report(run_dir, run_name, saved_cfg, CLASS_NAMES, history, metrics, setup, test_metrics)
        print(f'Saved report:  {report}')
    except Exception as e:  # the weights are already saved; the report can be rebuilt without retraining
        print(f'Writing the report failed ({type(e).__name__}: {e}). Rebuild it with:\n'
              f'  python make_report_detr.py --weights {weights_path}')


if __name__ == '__main__':
    main()
