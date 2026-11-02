"""Device selection and moving detection targets onto it."""
import torch


def get_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')  # rose laptops and the servers
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # mac
    print('Using device:', device)
    return device


def targets_to_device(targets, device):
    """Move a list of {'boxes', 'labels'} dicts to `device`."""
    return [{k: v.to(device) for k, v in t.items()} for t in targets]
