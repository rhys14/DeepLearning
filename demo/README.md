# Galaxy10 model comparison demo

Upload a galaxy image and the baseline CNN, VGG16, ResNet-18, DINOv2 and GalaxyDETR each classify it. One page
shows every model's prediction, its full confidence distribution over the 10 Galaxy10 DECaLS classes,
and the box GalaxyDETR drew around the galaxy.

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
| ResNet-18 | `galaxy_resnet18/resnet18_galaxy10_76pct.pth`: ImageNet-pretrained, layer3, layer4 and fc fine-tuned, 76% test accuracy |
| DINOv2 | `model_weights/best_dinov2.pth`: dinov2-small, last 6 transformer blocks and a linear head fine-tuned |
| GalaxyDETR | `model_weights/galaxy_detr_20261108_141353.pth`: ImageNet-pretrained MobileNetV4, random sampler, whole 256x256 image |

To show a different GalaxyDETR run, change `DETR_WEIGHTS` near the top of the model list in `app.py`.
Its label, preprocessing (whole image or 224 crop) and box placement are read from the config saved
inside the checkpoint. If `results/<run_name>/metrics.json` exists, its test accuracy is added to the label.

If a model's file is missing, the demo still runs and that model's column says which file to add.

## Demo images

Every image in `Galaxy_Images_For_Demo/` (PNG, JPEG, WebP, BMP or TIFF) appears in the sidebar. Each
one is run through all the models when the server starts, so clicking it shows its results instantly,
and the page opens on the first one. Each sidebar entry is labelled with its file name, without the
extension, underscores or copy suffixes like " (1)". Rename a file to change its label. Under the name
is the models' majority answer. Uploading, dropping or pasting your own image still works at any time
and clears the sidebar selection. To add or remove demo images, change the folder and restart the server.

## Random Galaxy10 images

If `data/Galaxy10_DECals.h5` exists, a "Random Galaxy10 image" button appears. It picks an image from
the dataset and marks the true label, so you can see which models got it right.

## How each model sees the image

Every upload is center-cropped to a square and resized to 256x256 first, so all four models see the
same galaxy and a Galaxy10 image passes through unchanged. Then each model gets the preprocessing it
was trained with:

- Baseline CNN and DINOv2: raw 256x256 pixels divided by 255, as in `Galaxy10Dataset`.
- VGG16 and ResNet-18: resize 256, center-crop 224, ImageNet normalization, as in the VGG notebook and `predict_galaxy.py`.
- GalaxyDETR: the whole 256x256 image with ImageNet normalization, as `prepare_image(img, crop=None)` does for this run.

For GalaxyDETR, the answer comes from the most confident of its 10 object queries. The distribution
shown is that query's softmax over the 10 classes, renormalized without the "no galaxy" slot.
