"""Presentation-quality confusion matrix for a GalaxyDETR run (run from the repo root).

    python plot_confusion_detr.py                          # latest run, test set
    python plot_confusion_detr.py --split validation
    python plot_confusion_detr.py --run galaxy_detr_20261108_141353
    python plot_confusion_detr.py --counts                 # raw image counts instead of row percentages
    python plot_confusion_detr.py --theme dark             # for dark slide backgrounds

Reads results/<run>/metrics.json (written by train_detr.py / make_report_detr.py), so no model or
GPU is needed. Saves results/<run>/confusion_<split>.png (300 dpi) and .svg (scales cleanly in slides).

Each cell shows the share of that TRUE class predicted as each class (rows sum to 100%), with the
image count underneath, so the diagonal is per-class recall and rare classes stay readable.
"""
import argparse
import glob
import json
import os

import numpy as np

# Display names for the 10 Galaxy10 DECaLS classes, in label order
PRETTY = {
    "disturbed": "Disturbed",
    "merging": "Merging",
    "round_smooth": "Round smooth",
    "in_between_round_smooth": "In-between smooth",
    "cigar_smooth": "Cigar smooth",
    "barred_spiral": "Barred spiral",
    "unbarred_tight_spiral": "Unbarred tight spiral",
    "unbarred_loose_spiral": "Unbarred loose spiral",
    "edge_on_no_bulge": "Edge-on, no bulge",
    "edge_on_with_bulge": "Edge-on, with bulge",
}

# Single-hue sequential blue ramp (light -> dark) and text/surface tokens per theme
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Light theme: low values near-white, high values dark navy. Dark theme gets its own steps from the
# same hue - low values sit dark near the surface and high values glow bright - not an automatic flip.
THEMES = {
    "light": {"bg": "#ffffff", "zero": "#f4f3f0", "ink": "#1f1e1c", "ink2": "#5f5e5a",
              "ramp": BLUE_RAMP, "ramp_start": 0.06},
    "dark":  {"bg": "#1c1c1a", "zero": "#262624", "ink": "#f2f1ee", "ink2": "#a9a8a3",
              "ramp": ["#1a2c45", "#163b6a"] + BLUE_RAMP[:0:-1][4:] + ["#cde2fb"], "ramp_start": 0.0},
}


def _luminance(rgba):
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def latest_run(results_dir):
    runs = sorted(d for d in glob.glob(os.path.join(results_dir, "galaxy_detr_*"))
                  if os.path.exists(os.path.join(d, "metrics.json")))
    if not runs:
        raise SystemExit(f"No runs with a metrics.json found in {results_dir}/")
    return runs[-1]  # names are galaxy_detr_YYYYmmdd_HHMMSS, so the last one is the newest


def load_split(metrics, split):
    """Returns (confusion matrix, accuracy, n_images, split name used)."""
    if metrics.get(split):
        s = metrics[split]
        return np.asarray(s["confusion_matrix"]), s["accuracy"], s["n_images"], split
    if split == "test" and metrics.get("validation"):
        print("This run has no test set; plotting the validation set instead.")
        return load_split(metrics, "validation")
    if "confusion_matrix" in metrics:  # very early report format (validation only)
        cm = np.asarray(metrics["confusion_matrix"])
        return cm, metrics.get("val_accuracy", np.trace(cm) / cm.sum()), int(cm.sum()), "validation"
    raise SystemExit(f"No '{split}' results in metrics.json")


def plot(cm, class_names, title, subtitle, out_base, theme="light", show_counts_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    t = THEMES[theme]
    n = len(class_names)
    names = [PRETTY.get(c, c.replace("_", " ").capitalize()) for c in class_names]
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_tot, out=np.zeros_like(cm, dtype=float), where=row_tot > 0)
    value = cm / cm.max() if show_counts_only else pct
    ramp = LinearSegmentedColormap.from_list("seq_blue", t["ramp"])
    r0 = t["ramp_start"]
    # cells and colorbar share one map
    cmap = LinearSegmentedColormap.from_list("seq_blue_cells", [ramp(r0 + (1 - r0) * x) for x in np.linspace(0, 1, 256)])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11.5, 9.0), facecolor=t["bg"])
    ax.set_facecolor(t["bg"])

    gap = 0.04  # thin surface-colored gap between cells
    for i in range(n):
        for j in range(n):
            v = value[i, j]
            face = t["zero"] if cm[i, j] == 0 else cmap(v)
            ax.add_patch(plt.Rectangle((j - 0.5 + gap / 2, i - 0.5 + gap / 2), 1 - gap, 1 - gap,
                                       facecolor=face, edgecolor="none"))
            if cm[i, j] == 0:
                continue
            dark_cell = _luminance(face) < 0.30   # pick the text color the cell can carry
            ink = "#ffffff" if dark_cell else "#1f1e1c"
            ink2 = "#d6e2f3" if dark_cell else "#3d3c39"
            if show_counts_only:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12,
                        fontweight="bold" if i == j else "normal", color=ink)
            else:
                p = pct[i, j] * 100
                ax.text(j, i - 0.13, "<1%" if p < 0.5 else f"{p:.0f}%", ha="center", va="center", fontsize=12.5,
                        fontweight="bold" if i == j else "normal", color=ink)
                ax.text(j, i + 0.22, f"{cm[i, j]}", ha="center", va="center", fontsize=8.5, color=ink2)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor", color=t["ink"], fontsize=10.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{nm}  (n={int(row_tot[i, 0])})" for i, nm in enumerate(names)],
                       color=t["ink"], fontsize=10.5)
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Predicted class", color=t["ink2"], fontsize=12, labelpad=12)
    ax.set_ylabel("True class", color=t["ink2"], fontsize=12, labelpad=12)

    # Slim colorbar keyed to the cell encoding
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100 if not show_counts_only else cm.max()))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["ink2"], length=0, labelsize=9.5)
    cb.set_label("Images (count)" if show_counts_only else "Share of true class (%)", color=t["ink2"], fontsize=10.5)

    fig.text(0.06, 0.97, title, ha="left", va="top", fontsize=18, fontweight="bold", color=t["ink"])
    fig.text(0.06, 0.928, subtitle, ha="left", va="top", fontsize=11.5, color=t["ink2"])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.89, bottom=0.19)

    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=300 if ext == "png" else None, facecolor=t["bg"])
    plt.close(fig)
    return [f"{out_base}.png", f"{out_base}.svg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run folder or name (default: the newest run in results/)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--counts", action="store_true", help="color and label by raw counts instead of row %%")
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--out", help="output path without extension (default: results/<run>/confusion_<split>)")
    args = ap.parse_args()

    if args.run:
        run_dir = args.run if os.path.isdir(args.run) else os.path.join(args.results_dir, args.run)
    else:
        run_dir = latest_run(args.results_dir)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    cm, acc, n_img, split = load_split(metrics, args.split)
    run_name = metrics.get("run_name", os.path.basename(run_dir))

    split_label = {"test": "held-out test set", "validation": "validation set"}[split]
    title = f"GalaxyDETR confusion matrix: {acc * 100:.2f}% accuracy"
    subtitle = (f"{n_img:,} images, {split_label}  ·  rows = true class, columns = predicted  ·  "
                + ("cells show image counts" if args.counts else "cells show % of each true class (diagonal = recall)"))
    suffix = "_counts" if args.counts else ""
    suffix += "_dark" if args.theme == "dark" else ""
    out_base = args.out or os.path.join(run_dir, f"confusion_{split}{suffix}")
    files = plot(cm, metrics["class_names"], title, subtitle, out_base, args.theme, args.counts)
    print(f"Run: {run_name}  ({split}, accuracy {acc:.4f})")
    for p in files:
        print("Saved:", p)


if __name__ == "__main__":
    main()
