"""Run a trained GalaxyDETR on new data and save the predictions to a CSV (run from the repo root).

    # image files (png/jpg/...), a folder of them, or a mix
    python predict_detr.py --weights models/galaxy_detr.pth --input my_images/ --out predictions.csv

    # an .h5 file with an "images" dataset of (N, H, W, 3) uint8, like Galaxy10
    python predict_detr.py --weights models/galaxy_detr.pth --input other_data.h5 --out predictions.csv

One row per image: the most confident query's class and score, its box (cx, cy, w, h normalized
to the 224x224 model input), and every query above --threshold (as JSON) for multi-galaxy images.

In your own code:
    from Networks.galaxydetr import load_galaxydetr
    from datasets.galaxy10_detr_dataset import prepare_image
    model, class_names = load_galaxydetr('models/galaxy_detr.pth', device)
    out = model(prepare_image(img)[None].to(device))   # img: PIL image or (H, W, 3) uint8 array
"""
import argparse
import csv
import glob
import json
import os

import h5py
import torch
from PIL import Image

from datasets.galaxy10_detr_dataset import prepare_image
from Networks.galaxydetr import load_galaxydetr, postprocess
from utils.device import get_device

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def iter_inputs(paths):
    """Yields (name, image) from image files, folders of images, and .h5 files."""
    for p in paths:
        if os.path.isdir(p):
            files = sorted(f for f in glob.glob(os.path.join(p, '*')) if f.lower().endswith(IMAGE_EXTS))
            for f in files:
                yield f, Image.open(f)
        elif p.lower().endswith(('.h5', '.hdf5')):
            with h5py.File(p, 'r') as f:
                for i in range(len(f['images'])):
                    yield f'{p}[{i}]', f['images'][i]
        else:
            yield p, Image.open(p)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='models/galaxy_detr.pth')
    ap.add_argument('--input', nargs='+', required=True, help='image files, folders, and/or .h5 files')
    ap.add_argument('--out', default='predictions.csv')
    ap.add_argument('--threshold', type=float, default=0.5, help='min score to list a detection')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    model, class_names = load_galaxydetr(args.weights, device)

    rows, names, batch = [], [], []

    def flush():
        out = model(torch.stack(batch).to(device))
        prob = out['pred_logits'].softmax(-1)[..., :-1]
        score, cls = prob.max(-1)
        best_q = score.argmax(-1)
        dets = postprocess(out, args.threshold)
        for k, name in enumerate(names):
            q = best_q[k]
            box = out['pred_boxes'][k, q].tolist()
            d = dets[k]
            rows.append({'image': name, 'class': class_names[cls[k, q]], 'score': f'{score[k, q]:.4f}',
                         'cx': f'{box[0]:.4f}', 'cy': f'{box[1]:.4f}', 'w': f'{box[2]:.4f}', 'h': f'{box[3]:.4f}',
                         'detections': json.dumps([{'class': class_names[l], 'score': round(s, 4),
                                                    'box': [round(v, 4) for v in b]}
                                                   for s, l, b in zip(d['scores'].tolist(), d['labels'].tolist(),
                                                                      d['boxes'].tolist())])})
        names.clear()
        batch.clear()

    for name, img in iter_inputs(args.input):
        names.append(name)
        batch.append(prepare_image(img))
        if len(batch) == args.batch_size:
            flush()
    if batch:
        flush()

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} predictions to {args.out}')


if __name__ == '__main__':
    main()
