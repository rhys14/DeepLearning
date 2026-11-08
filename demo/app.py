"""Galaxy10 model comparison demo.

Upload a galaxy image and see what the baseline CNN, VGG16, DINOv2 and GalaxyDETR each think it is,
with every model's full confidence distribution over the 10 Galaxy10 DECaLS classes.

Run from the repo root (the DeepLearning folder):
    python demo/app.py            then open http://localhost:8000
    python demo/app.py --port 8080 --device cuda

No web framework needed: the server is Python's standard library. The models need torch, torchvision,
transformers and timm, which are already in requirements.txt / requirements_detr.txt.

Weights it looks for (first match wins; a missing file just greys out that model's column):
    baseline CNN  model_weights/cnn_baseline.pth (or models/)
    VGG16         training/galaxy_classifier.pth
    DINOv2        checkpoints/best_dinov2.pth
    GalaxyDETR    one column per training run in results/galaxy_detr_*/, loading
                  model_weights/<run_name>.pth (or the run's results folder, or models/)
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the DeepLearning folder
sys.path.insert(0, ROOT)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

CLASS_NAMES = ["disturbed", "merging", "round_smooth", "in_between_round_smooth", "cigar_smooth",
               "barred_spiral", "unbarred_tight_spiral", "unbarred_loose_spiral",
               "edge_on_no_bulge", "edge_on_with_bulge"]  # label = index (0-9), same as training

# ---------------------------------------------------------------------------------------------
# Preprocessing, matched to how each model was trained
# ---------------------------------------------------------------------------------------------
# Galaxy10 DECaLS images are 256x256. Any upload is first center-cropped to a square and resized to
# 256x256, so a Galaxy10 image passes through untouched and every model sees the same galaxy.
def to_canonical(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img if img.size == (256, 256) else img.resize((256, 256), Image.BICUBIC)


# Baseline CNN and DINOv2 (Galaxy10Dataset): raw 256x256 pixels / 255, no normalization
raw_256 = torchvision.transforms.ToTensor()

# VGG notebook and GalaxyDETR (prepare_image): resize 256, center-crop 224, ImageNet normalization
imagenet_224 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------------------------
def _load_state(path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k.removeprefix("module."): v for k, v in sd.items()}  # strip DataParallel prefixes


def build_baseline(path, device):
    from Networks.baselinecnn import GalaxyBaseline
    model = GalaxyBaseline()
    model.load_state_dict(_load_state(path, device))
    return model


class VGGGalaxyModel(nn.Module):
    """Same layout as GalaxyModel in training/vgg_train_and_val.ipynb (so the state_dict keys match)."""
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=None)  # weights come from the checkpoint
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(nn.Linear(25088, len(CLASS_NAMES)))

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


def build_vgg(path, device):
    model = VGGGalaxyModel()
    model.load_state_dict(_load_state(path, device))
    return model


class DINOv2Wrapper(nn.Module):
    """Same layout as Networks/galaxyDinoV2.GalaxyDINOv2, built from the dinov2-small config directly
    so the server doesn't download the pretrained backbone just to overwrite it with the checkpoint."""
    def __init__(self):
        super().__init__()
        try:
            from transformers import Dinov2Config, Dinov2ForImageClassification
        except ImportError as e:
            import transformers
            raise RuntimeError(f"transformers {transformers.__version__} has no DINOv2 support. "
                               "Update it with: pip install -U transformers") from e
        cfg = Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6, mlp_ratio=4,
                           patch_size=14, image_size=518, num_labels=len(CLASS_NAMES))
        self.model = Dinov2ForImageClassification(cfg)

    def forward(self, x):
        return self.model(pixel_values=x).logits


# transformers renamed DINOv2's attention weights around v5. A checkpoint saved with one version has to
# be renamed to load into the other, so both directions are tried.
_NEW_TO_OLD = [(".attention.q_proj.", ".attention.attention.query."),
               (".attention.k_proj.", ".attention.attention.key."),
               (".attention.v_proj.", ".attention.attention.value."),
               (".attention.o_proj.", ".attention.output.dense.")]


def _rename(state, pairs):
    out = {}
    for k, v in state.items():
        for a, b in pairs:
            if a in k:
                k = k.replace(a, b)
                break
        out[k] = v
    return out


def _dinov2_key_variants(state):
    """The checkpoint may have been saved from GalaxyDINOv2 (model.dinov2.*), from the bare HF model
    (dinov2.*), with other prefixes, and with either transformers naming for the attention weights.
    Yield every combination to try against the wrapper's keys."""
    prefixed = [state, {"model." + k: v for k, v in state.items()}]
    for pre in ("model.model.", "backbone.", "net."):
        if any(k.startswith(pre) for k in state):
            prefixed.append({"model." + k.removeprefix(pre): v for k, v in state.items()})
    old_to_new = [(b, a) for a, b in _NEW_TO_OLD]
    for st in prefixed:
        yield st
        yield _rename(st, _NEW_TO_OLD)
        yield _rename(st, old_to_new)


def build_dinov2(path, device):
    state = _load_state(path, device)
    model = DINOv2Wrapper()
    target = set(model.state_dict())
    best = None
    for variant in _dinov2_key_variants(state):
        missing = target - set(variant)
        unexpected = set(variant) - target
        # position_ids buffers differ between transformers versions and don't hold weights
        missing = {k for k in missing if not k.endswith("position_ids")}
        unexpected = {k for k in unexpected if not k.endswith("position_ids")}
        if not missing and not unexpected:
            model.load_state_dict(variant, strict=False)
            return model
        if best is None or len(missing) + len(unexpected) < len(best[0]) + len(best[1]):
            best = (missing, unexpected)
    missing, unexpected = best
    raise RuntimeError(f"checkpoint keys don't match dinov2-small: {len(missing)} missing "
                       f"(e.g. {sorted(missing)[:2]}), {len(unexpected)} unexpected (e.g. {sorted(unexpected)[:2]})")


class DETRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def build_detr(path, device, cfg_path=None):
    """Self-contained checkpoints (train_detr.py) rebuild themselves. A bare state_dict from an older
    run is rebuilt from that run's results/<run>/config.yaml instead."""
    from Networks.galaxydetr import build_model
    from utils.checkpoint import load_checkpoint
    ckpt = load_checkpoint(path, device)
    cfg = ckpt["config"]
    if cfg is None:
        if not cfg_path or not os.path.isfile(cfg_path):
            raise ValueError("bare state_dict and no config.yaml to rebuild the architecture from")
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False  # the weights come from the checkpoint
    model = build_model(cfg, len(CLASS_NAMES))
    model.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()})
    return DETRWrapper(model)


def newest(pattern):
    files = glob.glob(os.path.join(ROOT, pattern))
    return max(files, key=os.path.getmtime) if files else None


MODELS = [
    {"key": "baseline", "name": "Baseline CNN", "detail": "5 conv layers, trained from scratch",
     "candidates": ["model_weights/cnn_baseline.pth", "models/cnn_baseline.pth", "models/baseline.pth", "checkpoints/cnn_baseline.pth",
                    "training/cnn_baseline.pth"],
     "build": build_baseline, "transform": raw_256},
    {"key": "vgg", "name": "VGG16", "detail": "ImageNet features, frozen, with a new linear head",
     "candidates": ["training/galaxy_classifier.pth", "models/galaxy_classifier.pth",
                    "checkpoints/galaxy_classifier.pth"],
     "build": build_vgg, "transform": imagenet_224},
    {"key": "dinov2", "name": "DINOv2", "detail": "dinov2-small backbone, frozen, with a linear head",
     "candidates": ["checkpoints/best_dinov2.pth", "models/best_dinov2.pth"],
     "build": build_dinov2, "transform": raw_256},
    # GalaxyDETR columns are added by discover_detr_runs(), one per results/galaxy_detr_* folder
]
FIXED_COLORS = {"baseline": "#7DB4E8", "vgg": "#E8B860", "dinov2": "#B59AF2"}
DETR_COLORS = ["#62D2B0", "#B3DB6A", "#F08FA8", "#E8E1C8"]


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def discover_detr_runs():
    """One GalaxyDETR model per training run in results/. Weights are looked for in the run's own
    results folder first, then where train_detr.py saves them (models/<run_name>.pth)."""
    import yaml
    runs = sorted(d for d in glob.glob(os.path.join(ROOT, "results", "galaxy_detr_*")) if os.path.isdir(d))
    specs = []
    for i, run_dir in enumerate(runs):
        run = os.path.basename(run_dir)
        cfg_path = os.path.join(run_dir, "config.yaml")
        candidates = [f"model_weights/{run}.pth", f"results/{run}/{run}.pth", f"glob:results/{run}/*.pth",
                      f"models/{run}.pth", f"checkpoints/{run}.pth"]
        # The config stored inside the checkpoint is what the run actually trained with, so it's used for
        # the label when the weights are here; results/<run>/config.yaml is only the fallback.
        cfg = None
        wpath = find_weights({"candidates": candidates})
        if wpath:
            try:
                cfg = torch.load(wpath, map_location="cpu", weights_only=True).get("config")
            except Exception:
                cfg = None
        if cfg is None:
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception:
                cfg = {}
        m, d = cfg.get("model", {}), cfg.get("data", {})
        pretrained = bool(m.get("pretrained"))
        weighted = d.get("sampler") == "weighted"
        test_acc = _read_json(os.path.join(run_dir, "metrics.json")).get("test", {}).get("accuracy")
        backbone = m.get("backbone", "mobilenetv4_conv_small").replace("mobilenetv4_conv_", "MobileNetV4 ").replace("_", " ")
        detail = (f"{backbone}, {'ImageNet-pretrained' if pretrained else 'trained from scratch'}, "
                  f"{'trained with a weighted random sampler (class-balanced batches)' if weighted else 'trained with a plain random sampler (natural class mix)'}")
        if test_acc is not None:
            detail += f". Test accuracy {test_acc * 100:.1f}%"
        stamp = run.removeprefix("galaxy_detr_")
        specs.append({
            "key": f"detr_{stamp}", "name": f"GalaxyDETR ({'weighted random' if weighted else 'random'} sampler)",
            "run": run, "detail": detail, "color": DETR_COLORS[i % len(DETR_COLORS)], "detr": True,
            "candidates": candidates,
            "missing_hint": f"Weights not found. Put {run}.pth in model_weights/",
            "build": lambda path, device, c=cfg_path: build_detr(path, device, c),
            "transform": imagenet_224,
        })
    # two runs with the same short name (e.g. both weighted): tell them apart by start time
    names = [s["name"] for s in specs]
    for s in specs:
        if names.count(s["name"]) > 1:
            t = s["run"].removeprefix("galaxy_detr_")
            s["name"] = s["name"][:-1] + f", {t[9:11]}:{t[11:13]})"
    if not specs:  # no results folders: fall back to the latest single checkpoint
        specs.append({"key": "detr", "name": "GalaxyDETR", "detail": "MobileNetV4 backbone, transformer encoder-decoder",
                      "color": DETR_COLORS[0], "detr": True,
                      "candidates": ["model_weights/galaxy_detr.pth", "models/galaxy_detr.pth", "glob:models/galaxy_detr_*.pth"],
                      "build": build_detr, "transform": imagenet_224})
    return specs


def find_weights(spec):
    for c in spec["candidates"]:
        p = newest(c[5:]) if c.startswith("glob:") else os.path.join(ROOT, c)
        if p and os.path.isfile(p):
            return p
    return None


def load_models(device):
    MODELS.extend(discover_detr_runs())
    for spec in MODELS:
        spec.setdefault("color", FIXED_COLORS.get(spec["key"], "#E6ECF3"))
        spec["path"] = find_weights(spec)
        spec["model"], spec["error"] = None, None
        if spec["path"] is None:
            spec["error"] = spec.get("missing_hint") or "Weights not found. Put the file at " + spec["candidates"][0]
            print(f"  {spec['name']:<26} missing  (looked for {spec['candidates'][0]})")
            continue
        try:
            spec["model"] = spec["build"](spec["path"], device).to(device).eval()
            print(f"  {spec['name']:<26} loaded   {os.path.relpath(spec['path'], ROOT)}")
        except Exception as e:
            spec["error"] = f"Couldn't load {os.path.relpath(spec['path'], ROOT)}: {type(e).__name__}: {e}"
            print(f"  {spec['name']:<26} FAILED   {spec['error']}")
            traceback.print_exc()


# ---------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------
LOCK = threading.Lock()  # one forward pass at a time keeps memory predictable


@torch.no_grad()
def run_all(img, device):
    canon = to_canonical(img)
    results = []
    for spec in MODELS:
        r = {"key": spec["key"], "name": spec["name"], "detail": spec["detail"], "color": spec["color"]}
        if spec["model"] is None:
            r.update(status="missing", message=spec["error"])
            results.append(r)
            continue
        try:
            x = spec["transform"](canon)[None].to(device)
            with LOCK:
                out = spec["model"](x)
        except Exception as e:
            traceback.print_exc()
            r.update(status="missing", message=f"Failed on this image: {type(e).__name__}: {e}")
            results.append(r)
            continue
        if spec.get("detr"):
            # The most confident query is the model's answer. Its softmax covers the 10 classes plus
            # "no galaxy"; the distribution shown is over the 10 classes, renormalized.
            prob = out["pred_logits"][0].softmax(-1)              # (Q, 11)
            q = prob[:, :-1].max(-1).values.argmax().item()
            p_obj = 1 - prob[q, -1].item()
            probs = (prob[q, :-1] / prob[q, :-1].sum()).tolist()
            cx, cy, w, h = out["pred_boxes"][0, q].tolist()       # normalized to the 224 crop
            # map from the 224x224 center crop back to the 256x256 image shown on the page
            box = [(16 + (cx - w / 2) * 224) / 256, (16 + (cy - h / 2) * 224) / 256, w * 224 / 256, h * 224 / 256]
            r.update(box=box, objectness=p_obj, query=q)
        else:
            probs = out[0].softmax(-1).tolist()
        top = max(range(len(probs)), key=probs.__getitem__)
        r.update(status="ok", probs=probs, pred=CLASS_NAMES[top], pred_index=top, confidence=probs[top])
        results.append(r)

    buf = io.BytesIO()
    canon.save(buf, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "classes": CLASS_NAMES, "results": results}


# Optional: random images from the Galaxy10 .h5, so the demo works without hunting for test images
H5_PATH = os.path.join(ROOT, "data", "Galaxy10_DECals.h5")


def random_sample():
    import h5py
    with h5py.File(H5_PATH, "r") as f:
        i = random.randrange(len(f["ans"]))
        return Image.fromarray(f["images"][i]), int(f["ans"][i]), i


# ---------------------------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    device = "cpu"

    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self.send_json({"classes": CLASS_NAMES, "has_samples": os.path.isfile(H5_PATH),
                            "models": [{"key": s["key"], "name": s["name"], "detail": s["detail"],
                                        "color": s["color"], "loaded": s["model"] is not None,
                                        "message": s["error"]}
                                       for s in MODELS]})
        elif self.path.startswith("/api/sample"):
            if not os.path.isfile(H5_PATH):
                return self.send_json({"error": "No dataset at data/Galaxy10_DECals.h5"}, 404)
            try:
                img, label, idx = random_sample()
                res = run_all(img, self.device)
                res.update(true_label=CLASS_NAMES[label], sample_index=idx)
                self.send_json(res)
            except Exception as e:
                traceback.print_exc()
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/api/predict":
            return self.send_json({"error": "Not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        if n == 0 or n > 40 * 1024 * 1024:
            return self.send_json({"error": "Send one image under 40 MB."}, 400)
        data = self.rfile.read(n)
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            return self.send_json({"error": "That file isn't an image the server can read. Try a PNG or JPEG."}, 400)
        try:
            self.send_json(run_all(img, self.device))
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": f"Inference failed: {type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser(description="Galaxy10 four-model comparison demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other machines on the network in")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # On Windows, Python's HTTP server can bind a port another server is already using, and the OLD
    # server keeps answering the browser. Refuse to start instead, so a stale demo can't hide changes.
    import socket
    with socket.socket() as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            print(f"Port {args.port} is already in use, probably by an older copy of this demo that's still running.\n"
                  f"Close that terminal window (or end its python.exe in Task Manager), or run with --port {args.port + 1}.")
            sys.exit(1)
    ThreadingHTTPServer.allow_reuse_address = False

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    try:
        import transformers, timm
        versions = f"torch {torch.__version__}, torchvision {torchvision.__version__}, transformers {transformers.__version__}, timm {timm.__version__}"
    except ImportError as e:
        versions = f"torch {torch.__version__} ({e})"
    print(f"Python {sys.version.split()[0]}, {versions}")
    print(f"Loading models on {device} from {ROOT}")
    load_models(device)
    Handler.device = device

    url = f"http://localhost:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nDemo running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
