"""Classification metrics shared across models (baseline CNN, VGG, GalaxyDETR)."""
import numpy as np


def accuracy(preds, trues):
    return float((np.asarray(preds) == np.asarray(trues)).mean())


def confusion_matrix(preds, trues, num_classes):
    """Rows = true class, columns = predicted class."""
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(trues, preds):
        confusion[t, p] += 1
    return confusion
