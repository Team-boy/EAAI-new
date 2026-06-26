#!/usr/bin/env python
"""Route3: 诊断结论跨检测器一致性 (hard-class 漏检/混淆 + 阈值余量, 3 个检测器)."""
import os, numpy as np
from ultralytics import YOLO, RTDETR

ROOT = "/home/fenghn/CvTest/data/Dataset B"
RUNS = "/home/fenghn/CvTest/experiments/runs"
NC, IOU_T = 21, 0.5
HARD = [7, 18, 10, 13, 17, 16, 11]               # 固定难类集(来自 yolov8m baseline R<0.35)
MODELS = [("YOLOv8m", "yolov8m_B"), ("YOLOv11m", "yolo11m_B"), ("RT-DETR-l", "rtdetr_B")]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def analyze(weights):
    Model = RTDETR if "rtdetr" in weights.lower() else YOLO
    m = Model(weights)
    same = {c: [] for c in range(NC)}; dec = {c: [0, 0, 0] for c in range(NC)}
    for r in m.predict(source=ROOT+"/images/val", conf=0.001, iou=0.6, stream=True, verbose=False, device=6):
        H, W = r.orig_shape
        stem = os.path.splitext(os.path.basename(r.path))[0]
        lbl = os.path.join(ROOT, "labels/val", stem + ".txt")
        gts = []
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5:
                    c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                    gts.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
        cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); bx = r.boxes.xyxy.cpu().numpy()
        for gc, gb in gts:
            sc, other25 = 0.0, False
            for k in range(len(cf)):
                if iou(bx[k], gb) >= IOU_T:
                    if cl[k] == gc:
                        sc = max(sc, float(cf[k]))
                    elif cf[k] >= 0.25:
                        other25 = True
            same[gc].append(sc)
            dec[gc][0 if sc >= 0.25 else (1 if other25 else 2)] += 1
    miss = np.mean([dec[c][2]/sum(dec[c]) for c in HARD if sum(dec[c])])
    conf = np.mean([dec[c][1]/sum(dec[c]) for c in HARD if sum(dec[c])])
    r25 = np.mean([np.mean(np.array(same[c]) >= 0.25) for c in HARD if same[c]])
    rmx = np.mean([np.mean(np.array(same[c]) > 0.0) for c in HARD if same[c]])
    return miss, conf, r25, rmx


print("hard-class (7,18,10,13,17,16,11) 诊断跨检测器一致性 [Dataset B val]")
print("  detector    missed%  confused%   R@0.25  R@max")
for name, run in MODELS:
    miss, conf, r25, rmx = analyze(f"{RUNS}/{run}/weights/best.pt")
    print(f"  {name:10}  {miss:.2f}     {conf:.2f}      {r25:.2f}    {rmx:.2f}")
print("\n>> 若 3 行都是 missed>>confused 且 R@max>>R@0.25 => 诊断结论检测器无关(普适性++)")
