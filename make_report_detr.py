"""Rebuild a training run's report from its saved checkpoint, without retraining (run from the repo root).

    python make_report_detr.py --weights models/galaxy_detr_20261105_104711.pth
    python make_report_detr.py --weights models/galaxy_detr_20261105_104711.pth --log train_output.txt

What it does:
  * loads the checkpoint (weights + the exact config the run used + class names),
  * rebuilds the same train/validation/test split from that config and re-scores validation and test
    (identical numbers to the end of training: same weights, same images, no randomness in evaluation),
  * takes the per-epoch losses from results/<run>/history.json if the run wrote one, otherwise from
    --log, a text file of the training terminal output (the "Epoch [i/N], Train Loss: ..." lines),
  * writes results/<run>/report.md, metrics.json, history.csv, loss_curve.png, config.yaml.

Override any config value if paths differ on this machine, e.g. data.h5_path=/scratch/Galaxy10_DECals.h5
"""
import argparse
import copy
import json
import os
import re
from datetime import datetime

import torch

from datasets.galaxy10_detr_dataset import build_dataloaders
from Networks.galaxydetr import build_model
from training.detr_validater import evaluate_model
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides
from utils.detr_report import write_report
from utils.device import get_device

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\], Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+)( \*)?(?: \((\d+)s\))?")
AVG_RE = re.compile(r"Average time per epoch: ([\d.]+)s")


def history_from_log(path):
    """Parse per-epoch losses from saved terminal output. Returns None if no epoch lines are found."""
    train, val, times, avg = {}, {}, {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                e = int(m.group(1))
                train[e], val[e] = float(m.group(3)), float(m.group(4))
                if m.group(6):
                    times[e] = float(m.group(6))
            a = AVG_RE.search(line)
            if a:
                avg = float(a.group(1))
    if not train:
        return None
    epochs = sorted(train)
    tl, vl = [train[e] for e in epochs], [val[e] for e in epochs]
    et = [times.get(e) for e in epochs] if times else []
    if avg is None and times:
        avg = sum(times.values()) / len(times)
    best = min(range(len(vl)), key=vl.__getitem__)
    return {"train_loss": tl, "val_loss": vl, "epoch_time": et, "best_epoch": epochs[best],
            "best_val_loss": vl[best], "avg_epoch_time": avg,
            "total_time": None if avg is None else avg * len(epochs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/galaxy_detr.pth", help="checkpoint saved by train_detr.py")
    ap.add_argument("--log", help="text file of the training terminal output (if the run has no history.json)")
    ap.add_argument("--out", help="output folder (default: results/<run_name>)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. data.h5_path=... data.num_workers=0")
    args = ap.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.weights, device)
    if ckpt["config"] is None:
        raise SystemExit(f"{args.weights} has no saved config (it predates self-contained checkpoints), "
                         "so its split and settings can't be reconstructed.")
    cfg = apply_overrides(ckpt["config"], args.overrides)
    report_cfg = copy.deepcopy(cfg)  # the run's real settings, for the report (cfg is modified below)
    class_names, info = ckpt["class_names"], ckpt["info"]
    run_name = info.get("run_name") or os.path.splitext(os.path.basename(args.weights))[0]
    run_dir = args.out or os.path.join(cfg.get("output", {}).get("results_dir", "results"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    print("Run:", run_name)

    # per-epoch history
    hist_json = os.path.join(run_dir, "history.json")
    history, source = None, None
    if os.path.exists(hist_json):
        with open(hist_json) as f:
            history, source = json.load(f), hist_json
    elif args.log:
        history, source = history_from_log(args.log), args.log
        if history is None:
            print(f"No 'Epoch [i/N], Train Loss: ...' lines found in {args.log}.")
    if history is None:
        history = {"train_loss": [], "val_loss": [], "epoch_time": [],
                   "best_epoch": info.get("best_epoch", -1), "best_val_loss": info.get("best_val_loss", float("nan")),
                   "avg_epoch_time": None, "total_time": None}
        print("No per-epoch history available: the report will have results but no loss curve or epoch table.")
    else:
        print(f"Per-epoch history from {source} ({len(history['train_loss'])} epochs)")
    # the checkpoint knows which epoch was actually saved
    history["best_epoch"] = info.get("best_epoch", history["best_epoch"])
    history["best_val_loss"] = info.get("best_val_loss", history["best_val_loss"])

    # re-score validation and test with the saved weights
    torch.manual_seed(cfg["seed"])
    cfg["model"]["pretrained"] = False
    train_loader, val_loader, test_loader = build_dataloaders(cfg)  # builds the split; no images loaded yet
    model = build_model(cfg, len(class_names)).to(device)
    metrics = evaluate_model(model, val_loader, device, args.weights, class_names, "Validation")
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate_model(model, test_loader, device, args.weights, class_names, "Test")

    m = re.search(r"(\d{8}_\d{6})", run_name)
    started = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M") if m else "unknown"
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    setup = {"started": started, "device": f"{device_name} (device used to rebuild this report)",
             "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "n_train": len(train_loader.dataset),
             "n_val": len(val_loader.dataset), "n_test": len(test_loader.dataset) if test_loader else 0,
             "weights_path": args.weights.replace(os.sep, "/"),
             "note": f"Report rebuilt with make_report_detr.py on {datetime.now():%Y-%m-%d %H:%M}. "
                     "Validation/test results were recomputed from the saved checkpoint"
                     + ("." if source else "; per-epoch losses were not available.")}
    report = write_report(run_dir, run_name, report_cfg, class_names, history, metrics, setup, test_metrics)
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()
