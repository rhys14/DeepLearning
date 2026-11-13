# Galaxy10 Morphology Classification

Five deep learning models that classify galaxy images from the **Galaxy10 DECaLS** dataset into 10 morphology classes, from a small CNN trained from scratch to **GalaxyDETR**, a compact DETR-style detector built for this project. A web demo runs all five on the same image side by side.

Course project for CSSE/MA 416 Deep Learning, Rose-Hulman Institute of Technology.

## Results

| Model | Approach | Parameters | Accuracy | Evaluated on |
|---|---|---|---|---|
| Baseline CNN | 5 conv layers + 3 fully connected layers, trained from scratch | 2.2M | not yet evaluated on held-out data | – |
| VGG16 | ImageNet-pretrained VGG16, frozen; new linear classifier | 138M (251K trained) | 64.06% | validation, stratified 80/20 split |
| ResNet-18 | ImageNet-pretrained ResNet-18; layer3, layer4 and classifier fine-tuned | 11.2M | 75.66% | test, 70/15/15 split |
| DINOv2 | `facebook/dinov2-small` vision transformer with a linear classification head | 22M | 85.15% | validation, 80/20 split |
| **GalaxyDETR** | MobileNetV4 backbone + transformer encoder-decoder, set-prediction loss; predicts class **and** bounding box | 2.7M (1.76M used, see [Known issues](#known-issues)) | **84.10%** | test, stratified 70/15/15 split |

The models were evaluated on different splits, so the numbers are close but not perfectly comparable. Each model's section says exactly how it was split.

## Dataset

[Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html) contains 17,736 color images (256 × 256 pixels) from the DESI Legacy Imaging Surveys, labeled by Galaxy Zoo volunteers into 10 classes:

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | Disturbed | 5 | Barred spiral |
| 1 | Merging | 6 | Unbarred tight spiral |
| 2 | Round smooth | 7 | Unbarred loose spiral |
| 3 | In-between round smooth | 8 | Edge-on, no bulge |
| 4 | Cigar-shaped smooth | 9 | Edge-on, with bulge |

The classes are imbalanced: cigar-shaped smooth galaxies are 1.9% of the dataset, round smooth galaxies 14.9%.

Download `Galaxy10_DECals.h5` (2.5 GB) from [Zenodo](https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5) and put it at `data/Galaxy10_DECals.h5`. The `data/` folder is gitignored. The ResNet-18 scripts instead download the same images from the Hugging Face dataset [`matthieulel/galaxy10_decals`](https://huggingface.co/datasets/matthieulel/galaxy10_decals).

## Repository layout

```
├── configs/
│   └── detr_config.yaml           GalaxyDETR hyperparameters, data and output settings
├── datasets/
│   ├── galaxy10_dataset.py        Galaxy10 loader used by the baseline CNN and DINOv2
│   └── galaxy10_detr_dataset.py   GalaxyDETR loader: split, augmentation, pseudo-boxes, memory-mapped images
├── Networks/
│   ├── baselinecnn.py             Baseline CNN
│   ├── galaxyDinoV2.py            DINOv2 classifier
│   ├── galaxydetr.py              GalaxyDETR model, loader and post-processing
│   └── detr_components.py         GalaxyDETR building blocks: backbones, transformer layers, positional encoding
├── training/
│   ├── trainer.py                 Baseline CNN training
│   ├── trainingDino2.py           DINOv2 training
│   ├── vgg_train_and_val.ipynb    VGG16 transfer learning notebook
│   ├── detr_trainer.py            GalaxyDETR training loop
│   ├── detr_loss.py               Hungarian matcher and set-prediction loss
│   └── detr_validater.py          GalaxyDETR evaluation
├── utils/                         Shared helpers: checkpoints, metrics, box operations, config loading, reports
├── galaxy_resnet18/               ResNet-18 training (run_galaxy.py), inference (predict_galaxy.py), results
├── Galaxy_ResNet18_Oliver/        Archive of the ResNet-18 experiments (frozen, layer3, weighted sampler)
├── demo/                          Web demo comparing all five models, plus Hugging Face Space packaging
├── Galaxy_Images_For_Demo/        Images shown in the demo's sidebar
├── model_weights/                 Trained checkpoints (baseline CNN, DINOv2, GalaxyDETR runs)
├── results/                       One folder per GalaxyDETR run: report, metrics, loss curve, confusion matrix
├── train_detr.py                  Train GalaxyDETR
├── evaluate_detr.py               Evaluate a GalaxyDETR checkpoint
├── predict_detr.py                Run GalaxyDETR on new images
├── make_report_detr.py            Rebuild a run's report without retraining
├── plot_confusion_detr.py         Presentation-quality confusion matrix for a run
├── requirements.txt               Shared dependencies
└── requirements_detr.txt          Shared dependencies + GalaxyDETR extras (timm, scipy, pyyaml)
```

## Setup

Python 3.11 or newer, with PyTorch installed for your machine ([pytorch.org](https://pytorch.org/get-started/locally/)). Then, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements_detr.txt
```

`requirements_detr.txt` includes `requirements.txt`, so this one command covers every model. The ResNet-18 scripts also need `pip install datasets`.

Run every command below from the repository root unless a section says otherwise.

## Models

### 1. Baseline CNN

A small convolutional network trained from scratch, as a reference point for the transfer-learning models.

- **Architecture** (`Networks/baselinecnn.py`): five 3×3 convolution layers (8 → 16 → 32 → 64 → 128 channels), each followed by ReLU and 2×2 max pooling, which shrinks a 256 × 256 image to an 8 × 8 × 128 feature map. Three fully connected layers (8192 → 256 → 128 → 10) produce the class scores.
- **Data**: `Galaxy10Dataset` scales pixels to [0, 1]; random 80/20 split (`torch.random_split`, seed 42).
- **Training** (`training/trainer.py`): Adam, learning rate 1e-3, batch size 64, 20 epochs, cross-entropy loss.

```bash
python -m training.trainer
```

### 2. VGG16 (transfer learning)

`training/vgg_train_and_val.ipynb` uses VGG16 pretrained on ImageNet as a fixed feature extractor.

- **Architecture**: VGG16's convolutional layers are frozen; its classifier is replaced by a single linear layer (25,088 → 10), so only 250,890 parameters are trained.
- **Data**: whole dataset, stratified 80/20 train/validation split (scikit-learn, seed 42). Images are resized to 256, center-cropped to 224 and normalized with ImageNet statistics.
- **Training**: Adam, learning rate 1e-4, batch size 32, 20 epochs; the checkpoint with the lowest validation loss is kept as `training/galaxy_classifier.pth`.
- **Result**: 64.06% validation accuracy. Validation loss bottoms out at epoch 8 and then rises while training loss keeps falling, a sign of overfitting.

Open the notebook in Jupyter and run the cells in order.

### 3. ResNet-18 (fine-tuning)

Scripts in `galaxy_resnet18/`; run them from inside that folder.

- **Architecture**: torchvision ResNet-18 pretrained on ImageNet, with a new 10-class final layer. `layer3`, `layer4` and the classifier are trained; earlier layers stay frozen.
- **Data**: 70/15/15 train/validation/test split of the Hugging Face copy of the dataset (seed 42). Training images get random crops, horizontal and vertical flips and random rotations; evaluation uses resize 256, center-crop 224, ImageNet normalization.
- **Training** (`run_galaxy.py`): up to 25 epochs with early stopping (patience 6) on validation loss.
- **Result**: 75.66% test accuracy for the checkpoint `resnet18_galaxy10_76pct.pth`. The archive in `Galaxy_ResNet18_Oliver/` holds the other experiments: fully frozen backbone (50.86%), layer3-onward fine-tuning (75.28%) and a weighted random sampler (78.33%).

```bash
cd galaxy_resnet18
python run_galaxy.py                     # train
python predict_galaxy.py path/to/image   # classify one image
```

### 4. DINOv2

`Networks/galaxyDinoV2.py` wraps Meta's self-supervised vision transformer `facebook/dinov2-small` (Hugging Face Transformers) with a linear classification head.

- **Architecture**: a ViT-S/14 backbone (12 transformer blocks, width 384, 6 heads) that splits the image into 14 × 14 patches, plus a linear head over its output. The code as committed freezes the backbone and trains only the head; the checkpoint used in the demo also fine-tunes the last 6 transformer blocks.
- **Data**: `Galaxy10Dataset`, random 80/20 split (seed 42).
- **Training** (`training/trainingDino2.py`): AdamW, learning rate 1e-3, weight decay 0.01, batch size 32, 20 epochs; the best validation-accuracy checkpoint is saved to `checkpoints/best_dinov2.pth`.
- **Result**: 85.15% validation accuracy.

```bash
python -m training.trainingDino2
```

### 5. GalaxyDETR

A compact version of DETR (Carion et al., 2020) that finds the galaxy in an image and classifies it. Instead of predicting a class directly, it predicts a small set of (class, bounding box) pairs and is trained with a set-matching loss.

**Architecture**

```
image 3×256×256
 → MobileNetV4-Conv-Small backbone (ImageNet-pretrained, output taken at stride 16) → 96×16×16
 → 1×1 conv to 128 channels, flatten to 256 tokens, + 2D sine positional encoding
 → transformer encoder, 3 layers (self-attention over all 256 tokens)
 → transformer decoder, 3 layers, with 10 learned object queries
 → per query: class head (10 classes + "no galaxy") and box head (cx, cy, w, h)
```

The transformer uses width 128, 4 attention heads, feed-forward width 512 and pre-norm layers. The backbone is fine-tuned at a 3× lower learning rate than the rest of the model.

**Training**

- **Matching:** the Hungarian algorithm pairs the ground-truth galaxy with one of the 10 predictions.
- **Loss:** cross-entropy + 5 × L1 + 2 × GIoU, with the "no galaxy" class down-weighted to 0.1.
- **Auxiliary losses:** the same loss is applied after every decoder layer.
- **Optimizer:** AdamW with cosine learning-rate decay, gradient clipping at 0.1, 50 epochs, batch size 32.
- **Boxes:** Galaxy10 has class labels but no boxes, so each image's box is a pseudo-label made by thresholding the central bright region and padding it by 10%.
- **Data:** stratified 70/15/15 train/validation/test split (seed 42).
- **Augmentation:** training images only get random flips and 90° rotations; validation and test images are never augmented.
- **Image loading:** images are converted once to a memory-mapped `.npy` file, so training never needs the whole dataset in RAM.

**Result** (run `galaxy_detr_20261108_141353`): **84.10% test accuracy**, 83.62% validation accuracy, mean box GIoU 0.922 against the pseudo-labels; about 107 s per epoch on an NVIDIA L40S. The full report is in `results/galaxy_detr_20261108_141353/report.md`.

**Usage**

```bash
python train_detr.py                                    # train with configs/detr_config.yaml
python train_detr.py training.epochs=20 data.sampler=weighted   # override any config value
python evaluate_detr.py --split test                    # test, val, both, train or all
python evaluate_detr.py --split all --save              # whole dataset + train/val/test breakdown
python predict_detr.py --input my_images/ --out predictions.csv
python plot_confusion_detr.py                           # confusion matrix figure for the latest run
python make_report_detr.py --weights model_weights/<run>.pth    # rebuild a run's report
```

Each training run is named after its start time. It writes its weights to `models/<run>.pth`, where the checkpoint stores the config and class names so the model can be rebuilt without the YAML file. It also writes a report folder to `results/<run>/` containing:

- `report.md`
- `metrics.json`
- `history.csv`
- `loss_curve.png`
- `config.yaml`

## Demo

`demo/app.py` serves a local web page where you upload a galaxy image, or pick one from `Galaxy_Images_For_Demo/`, and all five models classify it. It shows each model's prediction and confidence over the 10 classes, plus GalaxyDETR's bounding box.

```bash
python demo/app.py            # opens http://localhost:8000
```

On Windows you can double-click `demo/run_demo.bat`. `python demo/publish_space.py` publishes the demo as a public Hugging Face Space. See [`demo/README.md`](demo/README.md) for options and which weight file each model loads.

## Known issues

- **GalaxyDETR's parameter count includes unused layers.** timm builds the full MobileNetV4 network, but GalaxyDETR only uses its stride-16 output. The last two stages (944,096 parameters) still run on every forward pass, but they never affect a prediction or receive gradients. So 1,755,279 of the reported 2,699,375 parameters actually affect the output.
- **`training/trainer.py` fails on Linux as written.** It imports `networks.baselinecnn` while the folder is named `Networks`; Windows ignores the case difference, Linux does not. It also saves to an absolute path on one team member's server account (`/home/phelpsrw/...`), which needs changing before others can run it.
- **ResNet-18 and the local `datasets/` folder clash.** The ResNet-18 scripts import Hugging Face's `datasets` package. Run them from inside `galaxy_resnet18/`, because from the repository root the local `datasets/` folder shadows that package.

## AI usage statement

AI was used to assist the creation of all the models in this project: the baseline CNN, VGG16, ResNet-18, DINOv2 and GalaxyDETR, as well as the demo. AI tools helped write and debug code, design model architectures and training pipelines, explain concepts, and draft documentation, including this README. The team trained and evaluated every model, and all reported results come from the team's own runs.

## References

- N. Carion et al. *End-to-End Object Detection with Transformers.* ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- D. Qin et al. *MobileNetV4: Universal Models for the Mobile Ecosystem.* ECCV 2024. [arXiv:2404.10518](https://arxiv.org/abs/2404.10518)
- K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- M. Walmsley et al. *Galaxy Zoo DECaLS: Detailed Visual Morphology Measurements from Volunteers and Deep Learning for 314,000 Galaxies.* MNRAS 509:3966, 2022.
- Galaxy10 DECaLS dataset, [astroNN documentation](https://astronn.readthedocs.io/en/latest/galaxy10.html).
