#!/usr/bin/env python
"""(b) 验证: per-class 在不同 conf 阈值下的 recall 余量 (降阈值能否救回漏检类)."""
import glob, os, statistics, numpy as np
from ultralytics import YOLO

ROOT = "/home/fenghn/CvTest/data/Dataset B"
VIMG, VLBL = f"{ROOT}/images/val", f"{ROOT}/labels/val"
NC, IOU_T = 21, 0.5
m = YOLO("/home/fenghn/CvTest/experiments/runs/yolov8m_B/weights/best.pt")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


gt_conf = {c: [] for c in range(NC)}     # 每个 GT 被同类框命中的最高 conf (0=任何阈值都没命中)
for r in m.predict(source=VIMG, conf=0.001, iou=0.6, stream=True, verbose=False, device=2):
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
    cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); bx = r.boxes.xyxy.cpu().numpy()
    for gc, gb in gts:
        best = 0.0
        for k in range(len(cf)):
            if cl[k] == gc and iou(bx[k], gb) >= IOU_T and cf[k] > best:
                best = cf[k]
        gt_conf[gc].append(best)

rows = []
for c in range(NC):
    arr = np.array(gt_conf[c]); n = len(arr)
    if n == 0:
        continue
    R = lambda t: float((arr >= t).mean())
    rows.append((R(0.25), c, n, R(0.25), R(0.10), R(0.05), R(0.001)))
rows.sort()

print("Dataset B  per-class recall @ 不同 conf 阈值")
print("  cls   GT   R@.25  R@.10  R@.05  R@max   可恢复Δ(max-.25)")
for r25, c, n, a, b, d, mx in rows:
    print(f"  {c:<3} {n:>4}   {a:.2f}   {b:.2f}   {d:.2f}   {mx:.2f}    +{mx-a:.2f}")

hard = [x for x in rows if x[0] < 0.35]
print(f"\n难类(R@.25<0.35) {len(hard)}个: 平均 R@.25={statistics.mean(x[3] for x in hard):.2f}"
      f" -> R@max={statistics.mean(x[6] for x in hard):.2f}  (平均可恢复 +{statistics.mean(x[6]-x[0] for x in hard):.2f})")
print(">> 若 R@max >> R@.25 => 框存在只是被阈值滤掉, (b)成立; 若 R@max 也低 => 真没出框, 需(a)")
