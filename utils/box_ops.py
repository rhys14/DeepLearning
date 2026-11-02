"""Bounding-box helpers."""
from torchvision.ops import box_convert, generalized_box_iou  # noqa: F401  (re-exported)


def cxcywh_to_xyxy(b):
    return box_convert(b, 'cxcywh', 'xyxy')
