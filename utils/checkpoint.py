"""Self-contained GalaxyDETR checkpoints.

A checkpoint holds everything needed to rebuild the trained model on any machine, without the
config file:
    {"state_dict": weights and biases,
     "config":     the full config used for training (model architecture, data, loss, training),
     "class_names": the 10 Galaxy10 class names, in label order,
     "info":       run name, best epoch, best validation loss, ...}
Only plain Python types are stored, so torch.load works with weights_only=True.
"""
import torch


def save_checkpoint(path, model, cfg, class_names, **info):
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "class_names": list(class_names), "info": info}, path)


def load_checkpoint(path, device="cpu"):
    """Returns the checkpoint dict. A bare state_dict (older runs) comes back as {"state_dict": ...}."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in ckpt:
        ckpt = {"state_dict": ckpt, "config": None, "class_names": None, "info": {}}
    return ckpt
