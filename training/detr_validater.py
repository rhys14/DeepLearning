"""GalaxyDETR validation: loss on the validation split, and accuracy / box GIoU / confusion matrix
for a saved checkpoint."""
import torch

from utils.box_ops import cxcywh_to_xyxy, generalized_box_iou
from utils.checkpoint import load_checkpoint
from utils.device import targets_to_device
from utils.detr_metrics import accuracy, confusion_matrix


@torch.no_grad()
def validation_loss(model, criterion, loader, device):
    model.eval()
    total, n = 0.0, 0
    for images, targets in loader:
        loss, _ = criterion(model(images.to(device)), targets_to_device(targets, device))
        total += loss.item() * len(targets)
        n += len(targets)
    return total / n


@torch.no_grad()
def evaluate_model(model, loader, device, checkpoint_path, class_names, split_name="Validation"):
    """Image-level accuracy: the class of the most confident non-'no galaxy' query,
    so the number is directly comparable to the baseline CNN / VGG classifiers."""
    model.load_state_dict(load_checkpoint(checkpoint_path, device)["state_dict"])
    model.eval()
    preds, trues, ious = [], [], []
    for images, targets in loader:
        out = model(images.to(device))
        prob = out["pred_logits"].softmax(-1)[..., :-1]       # (B, Q, C)
        score, cls = prob.max(-1)                              # (B, Q)
        best_q = score.argmax(-1)                              # (B,)
        b = torch.arange(len(best_q))
        preds.append(cls[b, best_q].cpu())
        box = out["pred_boxes"][b, best_q].cpu()
        gt = torch.cat([t["boxes"] for t in targets])
        ious.append(torch.diag(generalized_box_iou(cxcywh_to_xyxy(box), cxcywh_to_xyxy(gt))))
        trues.append(torch.cat([t["labels"] for t in targets]))

    preds, trues = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    acc = accuracy(preds, trues)
    mean_giou = torch.cat(ious).mean().item()
    confusion = confusion_matrix(preds, trues, len(class_names))
    print(f"--- {split_name} set ({len(trues)} images) ---")
    print(f"{split_name} accuracy: {acc:.4f}")
    print(f"Mean GIoU of top box vs. pseudo-label box: {mean_giou:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):", class_names)
    print(confusion)
    return {"accuracy": acc, "mean_giou": mean_giou, "confusion": confusion.tolist(), "n_images": int(len(trues))}
