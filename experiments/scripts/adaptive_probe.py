# -*- coding: utf-8 -*-
"""自适应分辨率值不值得: oracle上界 + 简单触发器对比. 用已训模型纯推理."""
import os, glob
import numpy as np
import torch
from ultralytics import YOLO

ROOT = "/home/fenghn/CvTest/data/Dataset B"
SPLIT = "test2"
RUNS = "/home/fenghn/CvTest/experiments/runs"
W960 = f"{RUNS}/resB_v8m_960/weights/best.pt"
W1280 = f"{RUNS}/opt1_hires1280/weights/best.pt"
IOU_T = 0.5

m960 = YOLO(W960)
m1280 = YOLO(W1280)

imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
              for p in glob.glob(f"{ROOT}/images/{SPLIT}/{e}"))


def load_gt(stem):
    lbl = f"{ROOT}/labels/{SPLIT}/{stem}.txt"
    gts = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                gts.append((c, x, y, w, h))
    return gts


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def img_recall(preds, gts):
    """每图召回(类别需匹配). preds: list(cls,cx,cy,w,h,conf)."""
    if not gts:
        return None
    matched = set()
    for p in sorted(preds, key=lambda x: -x[5]):
        for gi, g in enumerate(gts):
            if gi not in matched and p[0] == g[0] and iou(p[1:5], g) >= IOU_T:
                matched.add(gi); break
    return len(matched)/len(gts)


def predict(model, path, imgsz):
    r = model.predict(path, imgsz=imgsz, conf=0.25, iou=0.6, verbose=False, device=2)[0]
    out = []
    cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
    H, W = r.orig_shape
    for b, c, cf1 in zip(r.boxes.xyxy.cpu().numpy(), cl, cf):
        cx = (b[0]+b[2])/2/W; cy = (b[1]+b[3])/2/H; w = (b[2]-b[0])/W; h = (b[3]-b[1])/H
        out.append((int(c), cx, cy, w, h, float(cf1)))
    return out


# 逐图: 960结果 / 1280结果 / GT召回 / 960最小框尺寸(触发信号)
r960_tot = r1280_tot = oracle_tot = 0
ngt = 0
n_1280_better = n_960_better = n_tie = 0
rows = []
for path in imgs:
    stem = os.path.splitext(os.path.basename(path))[0]
    gts = load_gt(stem)
    if not gts:
        continue
    p960 = predict(m960, path, 960)
    p1280 = predict(m1280, path, 1280)
    r960 = img_recall(p960, gts); r1280 = img_recall(p1280, gts)
    g = len(gts)
    r960_tot += r960*g; r1280_tot += r1280*g; oracle_tot += max(r960, r1280)*g; ngt += g
    if r1280 > r960: n_1280_better += 1
    elif r960 > r1280: n_960_better += 1
    else: n_tie += 1
    # 触发信号: 960预测的最小框面积(小→可能漏极小目标→该升1280)
    minA = min([w*h for _, _, _, w, h, _ in p960], default=1.0)
    rows.append((stem, r960, r1280, g, minA))

print("=== 逐图 960 vs 1280 (test2) ===")
print(f"  全960 recall = {r960_tot/ngt:.3f}")
print(f"  全1280 recall= {r1280_tot/ngt:.3f}")
print(f"  ORACLE(每图取优) = {oracle_tot/ngt:.3f}  <- 自适应理论上界")
print(f"  1280更好的图: {n_1280_better}, 960更好: {n_960_better}, 平手: {n_tie}")
print(f"  => oracle比全1280高 {oracle_tot/ngt - r1280_tot/ngt:+.3f}, 比全960高 {oracle_tot/ngt - r960_tot/ngt:+.3f}")

# 简单触发器: 按960最小框面积阈值决定是否用1280
print("\n=== 简单触发器(960最小框<阈值则用1280) ===")
for q in [0.3, 0.5, 0.7]:
    thrA = np.quantile([r[4] for r in rows], q)
    hit = used = 0
    for stem, r960, r1280, g, minA in rows:
        use1280 = minA < thrA
        r = r1280 if use1280 else r960
        hit += r*g; used += use1280
    print(f"  q={q} (用1280比例 {used}/{len(rows)}): recall={hit/ngt:.3f}")
