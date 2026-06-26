"""box→密度热图 GT 生成(纯函数)."""
import numpy as np
from scipy.ndimage import gaussian_filter


def boxes_to_heatmap(boxes, out_h, out_w, sigma=2.0):
    """YOLO 框列表 (cx,cy,w,h) 归一化 → [0,1] 密度热图 (out_h,out_w)."""
    hm = np.zeros((out_h, out_w), dtype=np.float32)
    for cx, cy, w, h in boxes:
        x1 = max(0, int(round((cx - w / 2) * out_w)))
        x2 = min(out_w, int(round((cx + w / 2) * out_w)))
        y1 = max(0, int(round((cy - h / 2) * out_h)))
        y2 = min(out_h, int(round((cy + h / 2) * out_h)))
        if x2 > x1 and y2 > y1:
            hm[y1:y2, x1:x2] = 1.0
    if sigma > 0 and hm.max() > 0:
        soft = gaussian_filter(hm, sigma=sigma)
        hm = np.maximum(hm, soft / soft.max() if soft.max() > 0 else soft)
        hm = np.clip(hm, 0.0, 1.0)
    return hm.astype(np.float32)
