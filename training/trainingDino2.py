
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from Networks.galaxyDinoV2 import GalaxyDINOv2
from datasets.galaxy10_dataset import Galaxy10Dataset


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# Model
model = GalaxyDINOv2(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    weight_decay=0.01
)

epochs = 20
batch_size = 32

# Dataset
dataset = Galaxy10Dataset("data/Galaxy10_DECals.h5")

# 80/20 split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# Training
Path("checkpoints").mkdir(exist_ok=True)

best_accuracy = 0.0

for epoch in range(epochs):
    model.train()

    # Keep frozen backbone in eval mode
    model.model.dinov2.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    val_loss /= val_total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/best_dinov2.pth"
        )

        print("Saved best DINOv2 model!")

print("Training complete!")




