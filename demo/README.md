# Galaxy10 model comparison demo

Upload a galaxy image and the baseline CNN, VGG16, DINOv2 and both GalaxyDETR runs each classify it. One page
shows every model's prediction, its full confidence distribution over the 10 Galaxy10 DECaLS classes,
and the box each GalaxyDETR run drew around the galaxy.

## Run it

From the repo root (the `DeepLearning` folder), with the environment you already train in:

    python demo/app.py

On Windows you can also double-click `demo/run_demo.bat`. The page opens at http://localhost:8000.

Options: `--port 8080`, `--device cuda`, `--host 0.0.0.0` (lets other machines on the network open it,
e.g. for a presentation), `--no-browser`.

You don't need any extra packages. The server uses Python's standard library, and the models use torch,
torchvision, transformers and timm, which `requirements_detr.txt` already installs.

## Weights

| Model | File it loads |
|---|---|
| Baseline CNN | `model_weights/cnn_baseline.pth` |
| VGG16 | `training/galaxy_classifier.pth` |
| DINOv2 | `checkpoints/best_dinov2.pth` |
| GalaxyDETR (random sampler) | `model_weights/galaxy_detr_20261105_111705.pth`: ImageNet-pretrained MobileNetV4, plain random sampling, 82.8% test accuracy |
| GalaxyDETR (weighted random sampler) | `model_weights/galaxy_detr_20261105_134718.pth`: ImageNet-pretrained MobileNetV4, weighted random sampling (class-balanced batches), 81.9% test accuracy |

There's one GalaxyDETR column per run folder in `results/galaxy_detr_*/`. The column's label comes from the
config saved inside the checkpoint, the test accuracy from the run's `metrics.json`, and its weights from
`model_weights/<run_name>.pth`. A new training run appears automatically once its results folder and
weights are both in place. `models/` and the run's own results folder also work as weight locations.

If a model's file is missing, the demo still runs and that model's column says which file to add.

## Random Galaxy10 images

If `data/Galaxy10_DECals.h5` exists, a "Random Galaxy10 image" button appears. It picks an image from
the dataset and marks the true label, so you can see which models got it right.

## How each model sees the image

Every upload is center-cropped to a square and resized to 256x256 first, so all four models see the
same galaxy and a Galaxy10 image passes through unchanged. Then each model gets the preprocessing it
was trained with:

- Baseline CNN and DINOv2: raw 256x256 pixels divided by 255, as in `Galaxy10Dataset`.
- VGG16 and GalaxyDETR: resize 256, center-crop 224, ImageNet normalization, as in the VGG notebook and `prepare_image`.

For GalaxyDETR, the answer comes from the most confident of its 10 object queries. The distribution
shown is that query's softmax over the 10 classes, renormalized without the "no galaxy" slot.
