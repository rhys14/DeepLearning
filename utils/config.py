"""Load a YAML config and apply command-line overrides."""
import yaml


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, overrides):
    """overrides: list like ['training.epochs=20', 'model.backbone=custom'].
    Values are parsed with YAML, so numbers/booleans/null come through typed."""
    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg
