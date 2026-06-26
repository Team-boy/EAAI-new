"""框 IoU 匹配算单类 recall(纯函数)."""


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def recall_at_iou(preds, gts, iou_thr=0.5):
    """preds: (cx,cy,w,h,conf); gts: (cx,cy,w,h). 无 GT 返回 None."""
    if not gts:
        return None
    matched = set()
    for p in sorted(preds, key=lambda x: -x[4]):
        for gi, g in enumerate(gts):
            if gi not in matched and _iou(p[:4], g) >= iou_thr:
                matched.add(gi)
                break
    return len(matched) / len(gts)
