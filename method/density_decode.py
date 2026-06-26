"""密度图→连通域→框(纯函数)."""
import numpy as np
from scipy.ndimage import label, find_objects


def heatmap_to_boxes(hm, thr=0.5, min_area=4):
    """hm (H,W)[0,1] → list (cx,cy,w,h,conf) 归一化; conf=块内均值."""
    H, W = hm.shape
    lbl, n = label(hm > thr)
    out = []
    for i, sl in enumerate(find_objects(lbl), start=1):
        if sl is None:
            continue
        blk = (lbl[sl] == i)
        if blk.sum() < min_area:
            continue
        ys, xs = sl
        conf = float(hm[sl][blk].mean())
        cx = ((xs.start + xs.stop) / 2) / W
        cy = ((ys.start + ys.stop) / 2) / H
        out.append((cx, cy, (xs.stop - xs.start) / W, (ys.stop - ys.start) / H, conf))
    return out
