"""GalaxyDETR: a compact DETR-style detector that finds and classifies galaxies.

image -> backbone (stride 16) -> 1x1 proj -> + sine pos-enc -> encoder
      -> N learned object queries -> decoder -> per query: class logits (+ "no galaxy") and box

Boxes are (cx, cy, w, h) normalized to [0, 1].
"""
import torch
import torch.nn as nn

from Networks.detr_components import build_backbone, MLP, sine_pos_embed, EncoderLayer, DecoderLayer


class GalaxyDETR(nn.Module):
    def __init__(self, num_classes, in_ch=3, d_model=128, heads=4, enc_layers=3,
                 dec_layers=3, ffn=512, num_queries=10, dropout=0.1,
                 backbone='mobilenetv4_conv_small', pretrained=True):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained, in_ch)
        self.proj = nn.Conv2d(self.backbone.out_channels, d_model, 1)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, heads, ffn, dropout) for _ in range(enc_layers))
        self.enc_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList(DecoderLayer(d_model, heads, ffn, dropout) for _ in range(dec_layers))
        self.dec_norm = nn.LayerNorm(d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # last index = "no galaxy"
        self.box_head = MLP(d_model, d_model, 4, 3)

    def forward(self, x):
        feat = self.proj(self.backbone(x))               # (B, d, H, W)
        B, d, H, W = feat.shape
        src = feat.flatten(2).transpose(1, 2)            # (B, HW, d)
        pos = sine_pos_embed(H, W, d, x.device)[None]    # (1, HW, d)

        for layer in self.encoder:
            src = layer(src, pos)
        memory = self.enc_norm(src)

        query_pos = self.query_embed.weight[None].expand(B, -1, -1)
        tgt = torch.zeros_like(query_pos)
        outs = []
        for layer in self.decoder:
            tgt = layer(tgt, memory, pos, query_pos)
            outs.append(self.dec_norm(tgt))
        hs = torch.stack(outs)                           # (L, B, Q, d)

        logits = self.class_head(hs)
        boxes = self.box_head(hs).sigmoid()
        out = {'pred_logits': logits[-1], 'pred_boxes': boxes[-1]}
        out['aux'] = [{'pred_logits': l, 'pred_boxes': b} for l, b in zip(logits[:-1], boxes[:-1])]
        return out


def build_model(cfg, num_classes):
    """Build GalaxyDETR from the `model` section of a config dict."""
    m = cfg['model']
    return GalaxyDETR(num_classes=num_classes, in_ch=m['in_ch'], d_model=m['d_model'],
                      heads=m['heads'], enc_layers=m['enc_layers'], dec_layers=m['dec_layers'],
                      ffn=m['ffn'], num_queries=m['num_queries'], dropout=m['dropout'],
                      backbone=m['backbone'], pretrained=m['pretrained'])


@torch.no_grad()
def postprocess(outputs, threshold=0.5):
    """Per-image list of dicts with scores, labels, boxes (cxcywh, normalized) above `threshold`."""
    prob = outputs['pred_logits'].softmax(-1)[..., :-1]
    scores, labels = prob.max(-1)
    res = []
    for s, l, b in zip(scores, labels, outputs['pred_boxes']):
        keep = s > threshold
        res.append({'scores': s[keep], 'labels': l[keep], 'boxes': b[keep]})
    return res
