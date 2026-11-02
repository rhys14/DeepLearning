"""DETR set loss for GalaxyDETR: Hungarian matching, then cross-entropy (with a down-weighted
"no galaxy" class) + L1 + GIoU, applied to the final decoder layer and every auxiliary layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.box_ops import cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """For each image, finds the one-to-one (query, target) assignment with the lowest
    cost = w_bbox * L1 - w_class * p(class) - w_giou * GIoU."""
    def __init__(self, w_class=2.0, w_bbox=5.0, w_giou=2.0):
        super().__init__()
        self.wc, self.wb, self.wg = w_class, w_bbox, w_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Q = outputs['pred_logits'].shape[:2]
        prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        pbox = outputs['pred_boxes'].flatten(0, 1)
        tcls = torch.cat([t['labels'] for t in targets])
        tbox = torch.cat([t['boxes'] for t in targets])
        if len(tcls) == 0:
            e = torch.empty(0, dtype=torch.long)
            return [(e, e) for _ in targets]
        C = (self.wb * torch.cdist(pbox, tbox, p=1)
             - self.wc * prob[:, tcls]
             - self.wg * generalized_box_iou(cxcywh_to_xyxy(pbox), cxcywh_to_xyxy(tbox)))
        C = C.view(B, Q, -1).cpu()
        sizes = [len(t['boxes']) for t in targets]
        idx = [linear_sum_assignment(c[i].numpy()) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)) for i, j in idx]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1, w_ce=1.0, w_bbox=5.0, w_giou=2.0):
        super().__init__()
        self.nc, self.matcher = num_classes, matcher
        self.w = {'ce': w_ce, 'bbox': w_bbox, 'giou': w_giou}
        weight = torch.ones(num_classes + 1)
        weight[-1] = eos_coef
        self.register_buffer('ce_weight', weight)

    def single(self, out, targets, n_boxes):
        idx = self.matcher(out, targets)
        b_idx = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(idx)])
        s_idx = torch.cat([s for s, _ in idx])
        tgt_cls = torch.full(out['pred_logits'].shape[:2], self.nc, dtype=torch.long,
                             device=out['pred_logits'].device)
        tgt_cls[b_idx, s_idx] = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, idx)])
        l_ce = F.cross_entropy(out['pred_logits'].transpose(1, 2), tgt_cls, self.ce_weight)

        src = out['pred_boxes'][b_idx, s_idx]
        tgt = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, idx)])
        l_bbox = F.l1_loss(src, tgt, reduction='sum') / n_boxes
        l_giou = (1 - torch.diag(generalized_box_iou(cxcywh_to_xyxy(src), cxcywh_to_xyxy(tgt)))).sum() / n_boxes
        total = self.w['ce'] * l_ce + self.w['bbox'] * l_bbox + self.w['giou'] * l_giou
        return total, {'ce': l_ce.item(), 'bbox': l_bbox.item(), 'giou': l_giou.item()}

    def forward(self, outputs, targets):
        n = max(sum(len(t['labels']) for t in targets), 1)
        total, logs = self.single(outputs, targets, n)
        for aux in outputs['aux']:
            total = total + self.single(aux, targets, n)[0]
        return total, logs


def build_criterion(cfg, num_classes):
    l = cfg['loss']
    matcher = HungarianMatcher(l['match_class'], l['match_bbox'], l['match_giou'])
    return SetCriterion(num_classes, matcher, l['eos_coef'], l['w_ce'], l['w_bbox'], l['w_giou'])
