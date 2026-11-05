"""Evaluate a trained GalaxyDETR checkpoint (run from the repo root).

    python evaluate_detr.py                                   # test set of the latest run (models/galaxy_detr.pth)
    python evaluate_detr.py --split all                       # ENTIRE dataset, plus the train/val/test breakdown
    python evaluate_detr.py --split all --save                # ...and save it to results/<run>/eval_all.md + .json
    python evaluate_detr.py --weights models/galaxy_detr_20261105_104711.pth --split both

--split: test (default) | val | both (val + test) | train | all (train + val + test = every image)

The split is rebuilt from the config saved INSIDE the checkpoint, so it always matches the run
that produced the weights. (Older checkpoints without a saved config fall back to --config.)
Training images are evaluated WITHOUT flips, exactly like validation and test.
"""
import argparse
import json
import os

import numpy as np
import torch

from datasets.galaxy10_detr_dataset import CLASS_NAMES, Galaxy10DETRDataset, build_dataloaders, collate
from Networks.galaxydetr import build_model
from training.detr_validater import evaluate_model
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides, load_config
from utils.device import get_device

SPLIT_LABEL = {'train': 'Training', 'val': 'Validation', 'test': 'Test', 'all': 'Entire dataset'}
SEEN = {'train': 'yes, trained on', 'val': 'used to pick the checkpoint', 'test': 'no, held out',
        'all': 'mostly (includes the training images)'}


def unaugmented(loader):
    """Same images as `loader` (e.g. the training split) but with augmentation off and no shuffling."""
    ds = loader.dataset
    plain = Galaxy10DETRDataset(ds.file_path, ds.indices, ds.labels, ds.npy_path, ds.images, train=False)
    return torch.utils.data.DataLoader(plain, batch_size=loader.batch_size, num_workers=loader.num_workers,
                                       collate_fn=collate, persistent_workers=loader.num_workers > 0)


def combine(results):
    """Merge per-split results into one result over all their images."""
    confusion = sum(np.asarray(r['confusion']) for r in results)
    n = sum(r['n_images'] for r in results)
    return {'accuracy': float(np.trace(confusion) / confusion.sum()),
            'mean_giou': sum(r['mean_giou'] * r['n_images'] for r in results) / n,
            'confusion': confusion.tolist(), 'n_images': int(n)}


def save_results(cfg, run_name, weights, split, results):
    out_dir = os.path.join(cfg.get('output', {}).get('results_dir', 'results'), run_name)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f'eval_{split}')
    with open(base + '.json', 'w') as f:
        json.dump({'run_name': run_name, 'weights': weights, 'class_names': CLASS_NAMES, 'results': results}, f, indent=2)
    L = [f'# GalaxyDETR evaluation: `{run_name}` ({split})\n', f'Weights: `{weights}`\n',
         '| Split | Images | Accuracy | Mean box GIoU | Seen in training? |', '|---|---|---|---|---|']
    for s, r in results.items():
        L.append(f"| {SPLIT_LABEL[s]} | {r['n_images']} | {r['accuracy']:.4f} | {r['mean_giou']:.4f} | {SEEN[s]} |")
    for s, r in results.items():
        L += ['', f'## {SPLIT_LABEL[s]}: confusion matrix (rows = true, columns = predicted)\n',
              '| true \\ pred | ' + ' | '.join(str(i) for i in range(len(CLASS_NAMES))) + ' |',
              '|---' * (len(CLASS_NAMES) + 1) + '|']
        for i, name in enumerate(CLASS_NAMES):
            L.append(f'| {i} {name} | ' + ' | '.join(str(v) for v in r['confusion'][i]) + ' |')
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/detr_config.yaml', help='only used for checkpoints without a saved config')
    ap.add_argument('--weights', help="checkpoint to evaluate (default: the config's `checkpoint`, i.e. the latest run)")
    ap.add_argument('--split', choices=['test', 'val', 'both', 'train', 'all'], default='test')
    ap.add_argument('--save', action='store_true', help='write results/<run>/eval_<split>.md and .json')
    ap.add_argument('overrides', nargs='*', help='config overrides, e.g. data.num_workers=0 data.h5_path=...')
    args = ap.parse_args()

    device = get_device()
    yaml_cfg = load_config(args.config)
    weights = args.weights or yaml_cfg['checkpoint']
    ckpt = load_checkpoint(weights, device)
    if ckpt['config'] is not None:
        cfg = ckpt['config']
        print(f'Using the split settings saved in {weights}')
    else:
        cfg = yaml_cfg
        print(f'{weights} has no saved config; using {args.config} (its split must match the training run)')
    cfg = apply_overrides(cfg, args.overrides)
    cfg['model']['pretrained'] = False  # weights come from the checkpoint; skip the ImageNet download
    run_name = ckpt['info'].get('run_name') or os.path.splitext(os.path.basename(weights))[0]

    torch.manual_seed(cfg['seed'])
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg, len(CLASS_NAMES)).to(device)

    split = args.split
    if test_loader is None and split in ('test', 'both'):
        print('No test set in this run (data.test_fraction is 0); evaluating the validation set.')
        split = 'val'
    wanted = {'test': ['test'], 'val': ['val'], 'both': ['val', 'test'], 'train': ['train'],
              'all': ['train', 'val', 'test']}[split]
    loaders = {'train': unaugmented(train_loader), 'val': val_loader, 'test': test_loader}

    results = {}
    for s in wanted:
        if loaders[s] is not None:
            results[s] = evaluate_model(model, loaders[s], device, weights, CLASS_NAMES, SPLIT_LABEL[s])

    if split == 'all':
        results['all'] = r = combine(list(results.values()))
        print(f"--- Entire dataset ({r['n_images']} images) ---")
        print(f"Entire-dataset accuracy: {r['accuracy']:.4f}")
        print(f"Mean GIoU of top box vs. pseudo-label box: {r['mean_giou']:.4f}")
        print('Confusion matrix (rows = true, cols = predicted):', CLASS_NAMES)
        print(np.asarray(r['confusion']))

    if len(results) > 1:
        print('\nSummary')
        print(f"{'Split':<16}{'Images':>8}{'Accuracy':>10}{'GIoU':>8}   Seen in training?")
        for s, r in results.items():
            print(f"{SPLIT_LABEL[s]:<16}{r['n_images']:>8}{r['accuracy']:>10.4f}{r['mean_giou']:>8.4f}   {SEEN[s]}")
        if 'train' in results and 'test' in results:
            gap = results['train']['accuracy'] - results['test']['accuracy']
            print(f"\nTrain - test accuracy gap: {gap:+.4f}  (a large gap means overfitting; report the TEST number)")

    if args.save:
        base = save_results(cfg, run_name, weights, split, results)
        print(f'\nSaved: {base}.md and {base}.json')


if __name__ == '__main__':
    main()
