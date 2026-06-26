# -*- coding: utf-8 -*-
"""VBAD v2 导出: 空间特征图(不GAP) + 低conf检测空间热图 + Type-V标签 + 检测缓存.
针对 head AUC=0.65 偏弱(GAP丢空间信息)的优化."""
import os, glob, pickle
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
OUT = "/home/fenghn/CvTest/method/vbad_cache2"
IOU_T = 0.5
FEAT_LAYER = 9          # SPPF 输出 (576ch, 20x20 @640)
GRID = 20               # 热图/特征空间分辨率

os.makedirs(OUT, exist_ok=True)
m640 = YOLO(W640); m1280 = YOLO(W1280)
feat_store = {}
m640.model.model[FEAT_LAYER].register_forward_hook(lambda m, i, o: feat_store.__setitem__("f", o.detach()))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, device=0)[0]
    W, H = img.size
    return [(int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H, float(cf))
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy())]


def hit(gt, preds):
    return any(p[0] == gt[0] and iou(gt[1:5], p[1:5]) >= IOU_T for p in preds)


def lowconf_heatmap(d_lo):
    """低conf检测(<0.25)的空间热图 GRIDxGRID: 每个格子累计低conf检测的(1-conf)."""
    hm = np.zeros((GRID, GRID), np.float32)
    for c, cx, cy, w, h, cf in d_lo:
        if cf < 0.25:
            gx, gy = min(GRID-1, int(cx*GRID)), min(GRID-1, int(cy*GRID))
            hm[gy, gx] += (1.0 - cf)
    return hm


def process(split):
    imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
                  for p in glob.glob(f"{ROOT}/images/{split}/{e}"))
    rows = []
    for path in imgs:
        stem = os.path.splitext(os.path.basename(path))[0]
        lbl = f"{ROOT}/labels/{split}/{stem}.txt"
        if not os.path.exists(lbl):
            continue
        gts = [(int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
               for p in (l.split() for l in open(lbl)) if len(p) >= 5]
        if not gts:
            continue
        im = Image.open(path).convert("RGB")
        feat_store.clear()
        d640 = detect(m640, im, 640, 0.25)
        import torch.nn.functional as _F
        feat_t = feat_store["f"].float()                          # (1,C,h,w)
        feat_t = _F.interpolate(feat_t, size=(GRID, GRID), mode="bilinear", align_corners=False)
        feat = feat_t.squeeze(0).cpu().numpy()                    # (C,GRID,GRID)
        feat_mean = feat.mean(0); feat_max = feat.max(0)          # (GRID,GRID) each
        d640_lo = detect(m640, im, 640, 0.05)
        hm = lowconf_heatmap(d640_lo)                              # (20,20)
        d1280 = detect(m1280, im, 1280, 0.25)
        typeV = int(any((not hit(g, d640)) and hit(g, d1280) for g in gts))
        spatial = np.stack([feat_mean, feat_max, hm]).astype(np.float32)  # (3,20,20)
        rows.append({"stem": stem, "spatial": spatial, "typeV": typeV,
                     "gts": gts, "d640": d640, "d1280": d1280})
    pickle.dump(rows, open(f"{OUT}/{split}.pkl", "wb"))
    pos = sum(r["typeV"] for r in rows)
    print(f"{split}: {len(rows)}图, Type-V {pos} ({pos/len(rows):.1%}), spatial {rows[0]['spatial'].shape}")


process("train2")
process("test2")
print("EXPORT2 DONE")
