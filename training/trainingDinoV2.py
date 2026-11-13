
import torch
import torch.nn as nn
from pathlib import Path

from Networks.galaxyDinoV2 import GalaxyDINOv2
from datasets.galaxy10_rotation_dataset import build_dataloaders


# -------------------------
# Configuration
# -------------------------

epochs = 20
batch_size = 32
num_classes = 10

backbone_lr = 1e-5
classifier_lr = 1e-4

checkpoint_path = Path("checkpoints/best_dinov2.pth")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}", flush=True)


# -------------------------
# Model
# -------------------------

model = GalaxyDINOv2(num_classes=num_classes).to(device)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel() for p in model.parameters()
)

print(
    f"Trainable parameters: {trainable_params:,} / {total_params:,}",
    flush=True
)


# -------------------------
# Loss and Optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.model.classifier.parameters(),
            "lr": classifier_lr
        },
        {
            "params": [
                p for p in model.model.dinov2.parameters()
                if p.requires_grad
            ],
            "lr": backbone_lr
        }
    ],
    weight_decay=0.01
)

# Gradually reduce learning rates
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)


# -------------------------
# Dataset
# -------------------------

train_loader, val_loader = build_dataloaders(
    fp="data/Galaxy10_DECals.h5",
    batch_size=batch_size,
    num_workers=4,
    in_memory=True
)


# -------------------------
# Training
# -------------------------

best_accuracy = -1.0

for epoch in range(epochs):

    model.train()

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

        # Prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0
        )

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


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:

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


    # -------------------------
    # Save Best Checkpoint
    # -------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Saved best DINOv2 model! "
            f"Validation accuracy: {best_accuracy:.2f}%",
            flush=True
        )

    scheduler.step()


# -------------------------
# Load Best Model
# -------------------------

print("\nTraining complete!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# -------------------------
# Confusion Matrix
# -------------------------

confusion_matrix = torch.zeros(
    (num_classes, num_classes),
    dtype=torch.int64
)

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        # Rows = actual class
        # Columns = predicted class
        indices = labels * num_classes + predictions

        counts = torch.bincount(
            indices.cpu(),
            minlength=num_classes * num_classes
        )

        confusion_matrix += counts.reshape(
            num_classes, num_classes
        )


# -------------------------
# Print Confusion Matrix
# -------------------------

print("\nCONFUSION MATRIX")
print("Rows = Actual Class")
print("Columns = Predicted Class\n")

print("Actual\\Pred", end=" ")

for i in range(num_classes):
    print(f"{i:>6}", end="")

print()

for i in range(num_classes):

    print(f"Class {i:<5}", end=" ")

    for j in range(num_classes):
        print(
            f"{confusion_matrix[i, j].item():>6}",
            end=""
        )

    print()


# -------------------------
# Per-Class Metrics
# -------------------------

print("\nPER-CLASS PERFORMANCE")
print(
    f"{'Class':<10}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Samples':>10}"
)

for i in range(num_classes):

    tp = confusion_matrix[i, i].item()

    actual_total = confusion_matrix[i, :].sum().item()
    predicted_total = confusion_matrix[:, i].sum().item()

    # Class accuracy = proportion of that class identified correctly
    class_accuracy = (
        tp / actual_total if actual_total > 0 else 0.0
    )

    precision = (
        tp / predicted_total if predicted_total > 0 else 0.0
    )

    recall = class_accuracy

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    print(
        f"{i:<10}"
        f"{class_accuracy * 100:>11.2f}%"
        f"{precision * 100:>11.2f}%"
        f"{recall * 100:>11.2f}%"
        f"{f1:>12.4f}"
        f"{actual_total:>10}"
    )


# -------------------------
# Overall Accuracy
# -------------------------

correct_predictions = confusion_matrix.diag().sum().item()
total_predictions = confusion_matrix.sum().item()

overall_accuracy = (
    100 * correct_predictions / total_predictions
)

print(f"\nFinal Best-Checkpoint Accuracy: {overall_accuracy:.2f}%")
