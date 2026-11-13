
import torch
import torch.nn as nn
from pathlib import Path

from Networks.galaxyDinoV2 import GalaxyDINOv2
from datasets.galaxy10_dataset import build_dataloaders


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}", flush=True)


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


# Dataset - load all images into RAM once
train_loader, test_loader = build_dataloaders(
    fp="data/Galaxy10_DECals.h5",
    batch_size=64,
    num_workers=4,
    in_memory=True
)


# Checkpoint directory
Path("checkpoints").mkdir(exist_ok=True)

best_accuracy = 0.0


# Training
for epoch in range(epochs):
    model.train()

    # Keep frozen DINOv2 backbone in eval mode
    model.model.dinov2.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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
                f"Loss: {loss.item():.4f}",
                flush=True
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

            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

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
        f"Val Acc: {val_accuracy:.2f}%",
        flush=True
    )

    # Save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/best_dinov2.pth"
        )

        print(
            f"Saved best DINOv2 model! "
            f"Validation accuracy: {best_accuracy:.2f}%",
            flush=True
        )

print("Training complete!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")



