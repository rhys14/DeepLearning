"""Galaxy10 DECaLS dataset for GalaxyDETR.

Three ways to read the images (config `data.image_source`):
  * "memmap" (default) -- the first run converts the .h5 once into an uncompressed .npy file next to it
    (~3.5 GB on disk, ~1 min, streamed in small blocks so it never needs 3.5 GB of RAM). Every process
    then memory-maps that file: the OS pages images in from disk as they are read and can drop them
    again under memory pressure, so RAM use stays small and nothing gets OOM-killed. After the first
    epoch the file is usually in the OS cache, so it is as fast as RAM.
  * "ram" -- load all images into RAM (~3.5 GB), like vgg_train_and_val.ipynb. Only if you have the memory.
  * "h5" -- lazy .h5 access like galaxy10_dataset.py. Low RAM but slow: the Galaxy10 .h5 is
    gzip-compressed in chunks of ~64 images, so reading ONE image at random decompresses the whole
    chunk (~60 ms/image, ~8 min/epoch).
Preprocessing follows vgg_train_and_val.ipynb. Split: stratified train / validation / test
(config data.val_fraction, data.test_fraction; default 70/15/15). Validation picks the best
checkpoint; the test set is only scored once, after training.

Input size (config `data.crop`): null = the whole 256x256 image (default); 224 = center-crop to
224x224 like vgg_train_and_val.ipynb. Configs/checkpoints from before this option existed have no
`crop` key and keep the old 224 crop, so older runs are still evaluated exactly as they were trained.

Training-only augmentation, applied on the fly when a TRAINING image is loaded:
  * data.augment_flips: random horizontal and vertical flip, each p = 0.5
  * data.augment_rot90: random rotation by 0, 90, 180 or 270 degrees (each p = 0.25)
Together they cover all 8 orientations of a square image. Both are exact pixel moves (no
interpolation, no black corners). Galaxies have no preferred orientation, so the class is unchanged,
and the pseudo-box is computed AFTER the transform, so it always matches the image.
Augmented images are never stored and never added to any split: the split is fixed by index before
training, and the validation/test datasets are built with train=False, so they always return their
own original, untransformed images.

Galaxy10 has a class label per image but NO bounding boxes. Every cutout is centered on its
galaxy, so `galaxy_box` derives one box per image from the central light blob (pseudo-label):
threshold at background median + 3*sigma (MAD estimate) -> connected components ->
take the component containing / nearest the center -> pad its bounding box by 10%.
"""
import os
import shutil
import time

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASS_NAMES = ["disturbed", "merging", "round_smooth", "in_between_round_smooth", "cigar_smooth",
               "barred_spiral", "unbarred_tight_spiral", "unbarred_loose_spiral",
               "edge_on_no_bulge", "edge_on_with_bulge"]  # label = index in this list (0-9)

LEGACY_CROP = 224  # what runs made before the `data.crop` option used


def make_geometry(crop=None):
    """Resize (shorter side) to 256, then optionally center-crop to `crop`. crop=None keeps the
    whole image. Geometry is kept separate from the tensor steps so the pseudo-box is computed on
    exactly the pixels the model sees."""
    steps = [torchvision.transforms.Resize(256)]
    if crop:
        steps.append(torchvision.transforms.CenterCrop(crop))
    return torchvision.transforms.Compose(steps)


geometry = make_geometry(LEGACY_CROP)  # kept for backward compatibility
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
    # Pick the galaxy's component with a rule that is symmetric under flips and 90-degree rotations,
    # so an augmented image always gets the same (transformed) box as the original:
    #   1. a component covering the central 2x2 pixels (largest if several), else
    #   2. the component closest to the exact image center ((H-1)/2, (W-1)/2), ties -> largest.
    sizes = np.bincount(comps.ravel(), minlength=n + 1)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    central = comps[int(np.floor(cy)):int(np.ceil(cy)) + 1, int(np.floor(cx)):int(np.ceil(cx)) + 1]
    labels_here = np.unique(central[central > 0])
    if len(labels_here):
        c = labels_here[np.argmax(sizes[labels_here])]
    else:
        yy, xx = np.indices((H, W))
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        ids = np.arange(1, n + 1)
        dmin = np.asarray(ndimage.minimum(dist, comps, ids))
        cand = ids[np.isclose(dmin, dmin.min())]
        c = cand[np.argmax(sizes[cand])]
    ys, xs = np.nonzero(comps == c)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx / W, cy / H, min(w / W, 1.0), min(h / H, 1.0)]


def prepare_image(img, crop=LEGACY_CROP):
    """Preprocess ONE image the same way as training/validation, for forward passes on new data.
    img: PIL image or (H, W, 3) uint8 array. Pass the crop the model was trained with:
    crop = model.config["data"].get("crop", 224) for a model from load_galaxydetr().
    Returns a (3, 256, 256) tensor for crop=None, (3, 224, 224) for crop=224."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return to_tensor(make_geometry(crop)(img.convert("RGB")))


def random_flips(img):
    """Random horizontal and vertical flip, each with p=0.5 (exact pixel moves, no interpolation)."""
    if torch.rand(1).item() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if torch.rand(1).item() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def random_rot90(img):
    """Random rotation by 0, 90, 180 or 270 degrees (exact pixel moves, no interpolation)."""
    k = int(torch.randint(0, 4, (1,)).item())
    if k == 1:
        img = img.transpose(Image.ROTATE_90)
    elif k == 2:
        img = img.transpose(Image.ROTATE_180)
    elif k == 3:
        img = img.transpose(Image.ROTATE_270)
    return img


def stratified_split(fp, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Train / validation / test indices, each with the same class mix. Only the labels are read.

    The test set is carved off FIRST, so it depends only on (test_fraction, seed): changing
    val_fraction never moves images into or out of the test set. test_fraction=0 gives no test
    set, and with val_fraction=0.20 reproduces the original 80/20 split exactly."""
    with h5py.File(fp, "r") as f:
        labels = f["ans"][:].astype(np.int64)
    idx = np.arange(len(labels))
    test_idx = np.array([], dtype=np.int64)
    if test_fraction > 0:
        idx, test_idx = train_test_split(idx, test_size=test_fraction, stratify=labels, random_state=seed)
    # val_fraction is a share of the WHOLE dataset, so rescale it to the remaining pool
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction / (1 - test_fraction),
                                          stratify=labels[idx], random_state=seed)
    for name, ids in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        if len(ids):
            print(name, ids.shape, "class counts:", np.bincount(labels[ids], minlength=len(CLASS_NAMES)))
    return train_idx, val_idx, test_idx


class Galaxy10DETRDataset(Dataset):
    """Returns (image, target): image (3, 224, 224) float32,
    target = {"boxes": (1, 4) cxcywh normalized, "labels": (1,)}.

    Image source, in priority order: `images` (an in-RAM array), `npy_path` (memory-mapped lazily in
    each worker), else the .h5 at `fp` (opened lazily in each worker). Open file handles and memmaps
    are never pickled into DataLoader workers (see __getstate__), so workers never copy the images."""
    def __init__(self, fp, indices=None, labels=None, npy_path=None, images=None, train=False,
                 crop=LEGACY_CROP, flips=True, rot90=False):
        """train=True enables augmentation (only ever set for the TRAINING split): `flips` and
        `rot90` choose which transforms. crop: None = whole image, 224 = center-crop."""
        self.file_path, self.npy_path = fp, npy_path
        self.train = train
        self.crop, self.flips, self.rot90 = crop, flips, rot90
        self.geometry = make_geometry(crop)
        self.images = images
        self.file = None   # lazily opened h5py.File
        self.mm = None     # lazily opened np.memmap
        if labels is None:
            with h5py.File(fp, "r") as f:
                labels = f["ans"][:].astype(np.int64)
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else indices

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        state["mm"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def _raw_image(self, j):
        if self.images is not None:
            return self.images[j]
        if self.npy_path is not None:
            if self.mm is None:
                self.mm = np.load(self.npy_path, mmap_mode="r")
            return np.array(self.mm[j])  # copy one image out of the map
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        return self.file["images"][j]

    def __getitem__(self, idx):
        j = int(self.indices[idx])
        img = self.geometry(Image.fromarray(self._raw_image(j)).convert("RGB"))
        if self.train:  # training split only; validation/test always get the original image
            if self.flips:
                img = random_flips(img)
            if self.rot90:
                img = random_rot90(img)
        target = {"boxes": torch.tensor([galaxy_box(img)], dtype=torch.float32),
                  "labels": torch.tensor([int(self.labels[j])])}
        return to_tensor(img), target


def collate(batch):
    """Images stack into a tensor; targets stay a list (each image can have a different object count)."""
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def default_npy_path(h5_path):
    return os.path.splitext(h5_path)[0] + "_images.npy"


def convert_h5_to_npy(h5_path, npy_path, block=512):
    """One-time conversion of the compressed .h5 images to an uncompressed, memory-mappable .npy.
    Streams `block` images at a time (~100 MB of RAM). Writes to a temp file and renames at the end,
    so an interrupted conversion is never mistaken for a finished one."""
    with h5py.File(h5_path, "r") as f:
        src = f["images"]
        n, shape = len(src), src.shape
        need = int(np.prod(shape)) + (1 << 20)
        out_dir = os.path.dirname(os.path.abspath(npy_path))
        free = shutil.disk_usage(out_dir).free
        if free < need:
            raise RuntimeError(f"Converting needs {need / 1e9:.1f} GB free in {out_dir}, but only "
                               f"{free / 1e9:.1f} GB is free. Set data.npy_path to a folder with more space "
                               "(e.g. a scratch directory), or use data.image_source=h5.")
        tmp = npy_path + ".partial"
        print(f"Converting {h5_path} -> {npy_path} (one time, {need / 1e9:.1f} GB)...", flush=True)
        t = time.time()
        # Plain sequential writes (not a writable memmap), so written data never piles up in RAM.
        with open(tmp, "wb") as out:
            np.lib.format.write_array_header_1_0(
                out, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
                      "fortran_order": False, "shape": tuple(shape)})
            for start in range(0, n, block):
                stop = min(start + block, n)
                out.write(np.ascontiguousarray(src[start:stop], dtype=np.uint8).tobytes())
                if (start // block) % 5 == 0 or stop == n:
                    print(f"  {stop}/{n} images ({time.time() - t:.0f}s)", flush=True)
    os.replace(tmp, npy_path)
    print(f"Conversion done in {time.time() - t:.0f}s", flush=True)


def ensure_npy(h5_path, npy_path=None):
    """Returns the path of a valid .npy copy of the images, converting first if needed."""
    npy_path = npy_path or default_npy_path(h5_path)
    with h5py.File(h5_path, "r") as f:
        expected = f["images"].shape
    if os.path.exists(npy_path):
        try:
            if np.load(npy_path, mmap_mode="r").shape == expected:
                print(f"Using memory-mapped images: {npy_path}", flush=True)
                return npy_path
        except ValueError:
            pass
        print(f"{npy_path} does not match {h5_path}; rebuilding it.", flush=True)
    convert_h5_to_npy(h5_path, npy_path)
    return npy_path


def load_into_memory(fp):
    """Read all images into RAM once (~3.5 GB)."""
    t = time.time()
    with h5py.File(fp, "r") as f:
        images = f["images"][:]
    print(f"Loaded {images.shape} into memory in {time.time() - t:.1f}s", flush=True)
    return images


def make_weighted_sampler(labels, power=1.0, seed=42):
    """WeightedRandomSampler over the TRAINING split that rebalances classes.

    Each image gets weight 1 / count(its class) ** power:
      power = 1.0 -> every class is drawn equally often (~10% each for Galaxy10)
      power = 0.5 -> halfway: rare classes boosted, but less repetition of the same few images
      power = 0   -> natural class mix (same as plain shuffling, but drawn with replacement)
    Draws len(labels) images per epoch WITH replacement, so rare-class images repeat within an
    epoch and some common-class images are skipped that epoch (they come back in later epochs).
    """
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    class_w = np.where(counts > 0, 1.0 / np.maximum(counts, 1) ** power, 0.0)
    weights = torch.as_tensor(class_w[labels], dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=gen)
    share = class_w * counts / (class_w * counts).sum()
    print("Weighted sampler (power %.2f): expected share of each training epoch per class:" % power)
    for name, n, p in zip(CLASS_NAMES, counts.astype(int), share):
        print(f"  {name:<24} {n:>6} images  natural {n / counts.sum():6.1%} -> sampled {p:6.1%}"
              f"  (each image ~{p * len(labels) / max(n, 1):.1f}x per epoch)")
    return sampler


def build_dataloaders(cfg):
    """Returns (train_loader, val_loader, test_loader); test_loader is None when data.test_fraction is 0.
    Only the training split is augmented and (with data.sampler: weighted) class-rebalanced;
    validation and test always keep the natural class mix."""
    d = cfg["data"]
    train_idx, val_idx, test_idx = stratified_split(d["h5_path"], d["val_fraction"],
                                                    d.get("test_fraction", 0.0), cfg["seed"])
    with h5py.File(d["h5_path"], "r") as f:
        labels = f["ans"][:].astype(np.int64)

    source = d.get("image_source", "memmap")
    images = npy_path = None
    if source == "memmap":
        npy_path = ensure_npy(d["h5_path"], d.get("npy_path"))
    elif source == "ram":
        images = load_into_memory(d["h5_path"])
    elif source != "h5":
        raise ValueError(f"data.image_source must be memmap, ram or h5, not {source!r}")

    common = dict(batch_size=cfg["training"]["batch_size"], num_workers=d["num_workers"], collate_fn=collate,
                  persistent_workers=d["num_workers"] > 0)
    crop = d.get("crop", LEGACY_CROP)  # missing key (older configs/checkpoints) -> the old 224 crop
    flips, rot90 = d.get("augment_flips", True), d.get("augment_rot90", False)
    # Only the training split is augmented. Validation and test are separate index sets built with
    # train=False: they never see an augmented image, and no augmented image is ever added to them.
    train_ds = Galaxy10DETRDataset(d["h5_path"], train_idx, labels, npy_path, images,
                                   train=flips or rot90, crop=crop, flips=flips, rot90=rot90)
    val_ds = Galaxy10DETRDataset(d["h5_path"], val_idx, labels, npy_path, images, train=False, crop=crop)
    test_loader = None
    if len(test_idx):
        test_ds = Galaxy10DETRDataset(d["h5_path"], test_idx, labels, npy_path, images, train=False, crop=crop)
        test_loader = DataLoader(test_ds, **common)
    if d.get("sampler", "weighted") == "weighted":
        sampler = make_weighted_sampler(labels[train_idx], d.get("sampler_power", 1.0), cfg["seed"])
        train_loader = DataLoader(train_ds, sampler=sampler, **common)  # sampler replaces shuffle=True
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common)
    return train_loader, DataLoader(val_ds, **common), test_loader
