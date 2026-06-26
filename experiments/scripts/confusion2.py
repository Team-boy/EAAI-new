#!/usr/bin/env python
"""手算 Dataset B 混淆矩阵: 预测↔GT 的 IoU 匹配, 区分'漏到背景' vs '类间混淆'."""
import glob, os, numpy as np
from ultralytics import YOLO

ROOT = "/home/fenghn/CvTest/data/Dataset B"
VIMG = f"{ROOT}/images/val"
VLBL = f"{ROOT}/labels/val"
NC = 21
IOU_T, CONF_T = 0.5, 0.25

m = YOLO("/home/fenghn/CvTest/experiments/runs/yolov8m_B/weights/best.pt")
cm = np.zeros((NC + 1, NC + 1))            # [pred, gt], 索引 NC = 背景


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


for r in m.predict(source=VIMG, conf=CONF_T, iou=0.6, stream=True, verbose=False, device=2):
    H, W = r.orig_shape
    stem = os.path.splitext(os.path.basename(r.path))[0]
    lbl = os.path.join(VLBL, stem + ".txt")
    gts = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                gts.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
    preds = [(int(cl), b) for b, cl in
             zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))]
    matched = set()
    for pc, pb in preds:                   # ultralytics 预测已按 conf 降序
        best, bj = IOU_T, -1
        for j, (gc, gb) in enumerate(gts):
            if j in matched:
                continue
            v = iou(pb, gb)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            matched.add(bj); cm[pc, gts[bj][0]] += 1
        else:
            cm[pc, NC] += 1                # 误报(背景被判成类)
    for j, (gc, gb) in enumerate(gts):
        if j not in matched:
            cm[NC, gc] += 1                # 漏检(GT 判成背景)

rows = []
for j in range(NC):
    tot = cm[:, j].sum()
    if tot == 0:
        continue
    correct = cm[j, j] / tot
    bg = cm[NC, j] / tot
    conf = sorted(((cm[i, j], i) for i in range(NC) if i != j), reverse=True)
    rows.append((correct, j, int(tot), bg, conf[0][1], conf[0][0] / tot))

rows.sort()
print("Dataset B 混淆分解 (正确率升序, 最难在上)")
print("  cls   GT    正确   漏到背景   最大混淆类(占比)")
for c, j, tot, bg, ti, tv in rows:
    print(f"  {j:<3}  {tot:>4}   {c:.2f}     {bg:.2f}      cls{ti}({tv:.2f})")

hard = [r for r in rows if r[0] < 0.35]
if hard:
    import statistics
    print(f"\n难类(正确率<0.35)共 {len(hard)} 个:")
    print(f"  平均 漏到背景      = {statistics.mean(r[3] for r in hard):.2f}")
    print(f"  平均 最大单类混淆  = {statistics.mean(r[5] for r in hard):.2f}")
    print("  >> 漏到背景 >> 混淆 => 纯漏检(走B); 混淆可观 => 类间混淆(走A)")
