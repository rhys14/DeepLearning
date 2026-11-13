
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from Networks.galaxyDinoV2 import GalaxyDINOv2
from datasets.galaxy10_rotation_dataset import build_dataloaders


# -------------------------
# Configuration
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint_path = "checkpoints/best_dinov2.pth"

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

class_names = {
    0: "Disturbed Galaxy",
    7: "Unbarred Loose Spiral"
}


# -------------------------
# Load Trained Model
# -------------------------

model = GalaxyDINOv2(num_classes=10).to(device)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print(f"Using device: {device}")
print("Loaded trained DINOv2 model.")


# -------------------------
# Load Validation Dataset
# -------------------------

_, val_loader = build_dataloaders(
    fp="data/Galaxy10_DECals.h5",
    batch_size=32,
    num_workers=4,
    seed=42,
    in_memory=True
)


# -------------------------
# Denormalize Images
# -------------------------

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def denormalize(image):
    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


# -------------------------
# Find Two Examples
# -------------------------

wrong_disturbed = None
correct_spiral = None

with torch.no_grad():

    for images, labels in val_loader:

        images_gpu = images.to(
            device, non_blocking=True
        )

        outputs = model(images_gpu)

        probabilities = torch.softmax(
            outputs, dim=1
        )

        predictions = probabilities.argmax(dim=1)

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            confidence = probabilities[
                i, predicted
            ].item() * 100

            # Actual disturbed, predicted loose spiral
            if (
                actual == 0
                and predicted == 7
                and wrong_disturbed is None
            ):
                wrong_disturbed = (
                    images[i].clone(),
                    actual,
                    predicted,
                    confidence
                )

            # Actual loose spiral, predicted loose spiral
            if (
                actual == 7
                and predicted == 7
                and correct_spiral is None
            ):
                correct_spiral = (
                    images[i].clone(),
                    actual,
                    predicted,
                    confidence
                )

        if (
            wrong_disturbed is not None
            and correct_spiral is not None
        ):
            break


# -------------------------
# Verify Examples Found
# -------------------------

if wrong_disturbed is None:
    raise RuntimeError(
        "No disturbed galaxy predicted as "
        "unbarred loose spiral was found."
    )

if correct_spiral is None:
    raise RuntimeError(
        "No correctly classified "
        "unbarred loose spiral was found."
    )


# -------------------------
# Plot Comparison
# -------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 6)
)

examples = [
    wrong_disturbed,
    correct_spiral
]

titles = [
    "Incorrectly Classified Disturbed Galaxy",
    "Correctly Classified Unbarred Loose Spiral"
]

colors = ["red", "green"]

for ax, example, title, color in zip(
    axes, examples, titles, colors
):

    image, actual, predicted, confidence = example

    ax.imshow(denormalize(image))

    ax.set_title(
        f"{title}\n"
        f"Actual: {class_names[actual]}\n"
        f"Predicted: {class_names[predicted]}\n"
        f"Confidence: {confidence:.1f}%",
        color=color,
        fontsize=11
    )

    ax.axis("off")


plt.suptitle(
    "DINOv2 Galaxy Classification Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    output_dir / "galaxy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved comparison to "
    "results/galaxy_comparison.png"
)
