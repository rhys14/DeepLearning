"""Galaxy10 DECaLS dataset for GalaxyDETR.

Same lazy .h5 access as galaxy10_dataset.py (the file is opened on first use inside each
DataLoader worker, so the ~3.5 GB of images never sit in RAM), with the preprocessing and
stratified 80/20 split from vgg_train_and_val.ipynb.

Galaxy10 has a class label per image but NO bounding boxes. Every cutout is centered on its
galaxy, so `galaxy_box` derives one box per image from the central light blob (pseudo-label):
threshold at background median + 3*sigma (MAD estimate) -> connected components ->
take the component containing / nearest the center -> pad its bounding box by 10%.
"""
import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["disturbed", "merging", "round_smooth", "in_between_round_smooth", "cigar_smooth",
               "barred_spiral", "unbarred_tight_spiral", "unbarred_loose_spiral",
               "edge_on_no_bulge", "edge_on_with_bulge"]  # label = index in this list (0-9)

# Geometry steps are split from the tensor steps so the box is computed on exactly the
# pixels the model sees (256 -> center-crop 224). Combined, this is the VGG notebook's pipeline.
geometry = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])
to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def galaxy_box(img, nsigma=3.0, pad=0.10, fallback=0.3):
    """Box (cx, cy, w, h), normalized, around the central galaxy of an (H, W, 3) image."""
    gray = np.asarray(img, dtype=np.float32).mean(axis=2)
    H, W = gray.shape
    med = np.median(gray)
    sigma = 1.4826 * np.median(np.abs(gray - med)) + 1e-6
    mask = ndimage.binary_opening(gray > med + nsigma * sigma, iterations=1)
    comps, n = ndimage.label(mask)
    if n == 0:
        return [0.5, 0.5, fallback, fallback]
    c = comps[H // 2, W // 2]
    if c == 0:  # center pixel not above threshold: use the component nearest the center
        ys, xs = np.nonzero(comps)
        k = np.argmin((ys - H / 2) ** 2 + (xs - W / 2) ** 2)
        c = comps[ys[k], xs[k]]
    ys, xs = np.nonzero(comps == c)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx / W, cy / H, min(w / W, 1.0), min(h / H, 1.0)]


def stratified_split(fp, val_fraction=0.20, seed=42):
    """Train/validation indices with the same class mix in both. Only the labels are read."""
    with h5py.File(fp, "r") as f:
        labels = f["ans"][:].astype(np.int64)
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=val_fraction,
                                          stratify=labels, random_state=seed)
    for name, idx in [("train", train_idx), ("validation", val_idx)]:
        print(name, idx.shape, "class counts:", np.bincount(labels[idx], minlength=len(CLASS_NAMES)))
    return train_idx, val_idx


class Galaxy10DETRDataset(Dataset):
    """Returns (image, target): image (3, 224, 224) float32,
    target = {"boxes": (1, 4) cxcywh normalized, "labels": (1,)}."""
    def __init__(self, fp, indices=None):
        self.file_path = fp
        self.file = None
        if indices is None:
            with h5py.File(fp, "r") as f:
                indices = np.arange(len(f["images"]))
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        j = int(self.indices[idx])
        img = geometry(Image.fromarray(self.file["images"][j]).convert("RGB"))
        target = {"boxes": torch.tensor([galaxy_box(img)], dtype=torch.float32),
                  "labels": torch.tensor([int(self.file["ans"][j])])}
        return to_tensor(img), target


def collate(batch):
    """Images stack into a tensor; targets stay a list (each image can have a different object count)."""
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def build_dataloaders(cfg):
    d = cfg["data"]
    train_idx, val_idx = stratified_split(d["h5_path"], d["val_fraction"], cfg["seed"])
    common = dict(batch_size=cfg["training"]["batch_size"], num_workers=d["num_workers"], collate_fn=collate)
    train_loader = DataLoader(Galaxy10DETRDataset(d["h5_path"], train_idx), shuffle=True, **common)
    val_loader = DataLoader(Galaxy10DETRDataset(d["h5_path"], val_idx), **common)
    return train_loader, val_loader
