"""Write a training run's results to a folder for documentation.

results/<run_name>/
    report.md        human-readable summary: setup, best epoch, accuracy, confusion matrix, per-epoch losses
    metrics.json     the same numbers, machine-readable (for comparing runs in code)
    history.csv      per-epoch train loss, validation loss, epoch time
    loss_curve.png   training and validation loss over epochs
    config.yaml      the exact config the run used
"""
import csv
import json
import os

import numpy as np
import yaml


def _per_class_rows(confusion, class_names):
    rows = []
    for i, name in enumerate(class_names):
        n_true = int(confusion[i].sum())
        n_pred = int(confusion[:, i].sum())
        tp = int(confusion[i, i])
        recall = tp / n_true if n_true else float("nan")
        precision = tp / n_pred if n_pred else float("nan")
        rows.append({"class": name, "support": n_true, "precision": precision, "recall": recall})
    return rows


def _fmt(x):
    return "n/a" if x != x else f"{x:.3f}"  # x != x catches NaN


def _plot_losses(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    epochs = list(range(len(history["train_loss"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.axvline(history["best_epoch"], color="gray", linestyle="--", linewidth=1,
               label=f"Best epoch ({history['best_epoch']})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DETR loss")
    ax.set_title("Training and Validation Loss Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(run_dir, run_name, cfg, class_names, history, metrics, setup):
    """history: from train_model; metrics: from evaluate_model; setup: device, params, split sizes, paths."""
    os.makedirs(run_dir, exist_ok=True)
    confusion = np.asarray(metrics["confusion"])
    per_class = _per_class_rows(confusion, class_names)

    # config.yaml
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # history.csv
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "epoch_time_s", "best_so_far"])
        best = float("inf")
        for e, (tr, va, t) in enumerate(zip(history["train_loss"], history["val_loss"], history["epoch_time"])):
            is_best = va < best
            best = min(best, va)
            w.writerow([e, f"{tr:.6f}", f"{va:.6f}", f"{t:.1f}", "*" if is_best else ""])

    # loss_curve.png
    _plot_losses(history, os.path.join(run_dir, "loss_curve.png"))

    # metrics.json
    summary = {
        "run_name": run_name,
        "setup": setup,
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "epochs_run": len(history["train_loss"]),
        "avg_epoch_time_s": history["avg_epoch_time"],
        "total_train_time_s": history["total_time"],
        "val_accuracy": metrics["accuracy"],
        "mean_box_giou": metrics["mean_giou"],
        "class_names": list(class_names),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # report.md
    m, t, d = cfg["model"], cfg["training"], cfg["data"]
    L = []
    L.append(f"# GalaxyDETR training report: `{run_name}`\n")
    L.append("## Results\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Validation accuracy | **{metrics['accuracy']:.4f}** |")
    L.append(f"| Mean box GIoU (top query vs. pseudo-label box) | {metrics['mean_giou']:.4f} |")
    L.append(f"| Best epoch | {history['best_epoch']} of {len(history['train_loss'])} (0-indexed) |")
    L.append(f"| Best validation loss | {history['best_val_loss']:.5f} |")
    L.append(f"| Average time per epoch | {history['avg_epoch_time']:.1f} s |")
    L.append(f"| Total training time | {history['total_time'] / 60:.1f} min |")
    L.append(f"| Saved weights | `{setup['weights_path']}` |")
    L.append("")
    L.append("![Loss curve](loss_curve.png)\n")
    L.append("## Setup\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Date | {setup['started']} |")
    L.append(f"| Device | {setup['device']} |")
    L.append(f"| Dataset | `{d['h5_path']}` |")
    L.append(f"| Train / validation images | {setup['n_train']} / {setup['n_val']} (stratified, seed {cfg['seed']}) |")
    L.append(f"| Training flips | {'on' if d.get('augment_flips', True) else 'off'} |")
    L.append(f"| Backbone | {m['backbone']} ({'ImageNet-pretrained' if m['pretrained'] else 'from scratch'}) |")
    L.append(f"| Transformer | d_model {m['d_model']}, {m['heads']} heads, {m['enc_layers']} enc + "
             f"{m['dec_layers']} dec layers, FFN {m['ffn']}, {m['num_queries']} queries |")
    L.append(f"| Trainable parameters | {setup['n_params']:,} |")
    L.append(f"| Epochs / batch size | {t['epochs']} / {t['batch_size']} |")
    L.append(f"| Learning rate (backbone / rest) | {t['lr_backbone']} / {t['lr']} |")
    L.append("")
    L.append("## Per-class results (validation)\n")
    L.append("| Class | Images | Precision | Recall |\n|---|---|---|---|")
    for r in per_class:
        L.append(f"| {r['class']} | {r['support']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} |")
    L.append("")
    L.append("## Confusion matrix (rows = true class, columns = predicted class)\n")
    idx = " | ".join(str(i) for i in range(len(class_names)))
    L.append(f"| true \\ pred | {idx} |")
    L.append("|---" * (len(class_names) + 1) + "|")
    for i, name in enumerate(class_names):
        L.append(f"| {i} {name} | " + " | ".join(str(v) for v in confusion[i]) + " |")
    L.append("")
    L.append("## Per-epoch losses\n")
    L.append("| Epoch | Train loss | Val loss | Time (s) | |\n|---|---|---|---|---|")
    best = float("inf")
    for e, (tr, va, tm) in enumerate(zip(history["train_loss"], history["val_loss"], history["epoch_time"])):
        mark = "best so far" if va < best else ""
        best = min(best, va)
        L.append(f"| {e} | {tr:.5f} | {va:.5f} | {tm:.0f} | {mark} |")
    L.append("")
    L.append("Full config: [`config.yaml`](config.yaml). Machine-readable numbers: [`metrics.json`](metrics.json).\n")
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return os.path.join(run_dir, "report.md")
