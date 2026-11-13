import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class Galaxy10Dataset(Dataset):
    def __init__(self, fp, in_memory=True, transform=None):
        self.fp = fp
        self.transform = transform
        self.in_memory = in_memory
        self.h5 = None

        with h5py.File(fp, "r") as f:
            self.labels = f["ans"][:].astype(np.int64)

            if in_memory:
                print("Loading Galaxy10 images into RAM...")
                self.images = f["images"][:]
                print("Finished loading images into RAM.")
            else:
                self.images = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
        else:
            if self.h5 is None:
                self.h5 = h5py.File(self.fp, "r")
            image = self.h5["images"][idx]

        label = int(self.labels[idx])

        # Image shape: (H, W, C)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        image = image.float() / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataloaders(
    fp,
    batch_size=64,
    num_workers=4,
    seed=42,
    in_memory=True
):
    # DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = Galaxy10Dataset(
        fp=fp,
        in_memory=in_memory,
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader