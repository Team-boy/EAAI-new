#!/usr/bin/env python
"""(b) 方法: per-class 自适应阈值. train 调阈值(F2-max), val 评测, 对比统一 0.25."""
import os, numpy as np
from ultralytics import YOLO

ROOT = "/home/fenghn/CvTest/data/Dataset B"
NC, IOU_T = 21, 0.5
m = YOLO("/home/fenghn/CvTest/experiments/runs/yolov8m_B/weights/best.pt")


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def collect(split):
    img_dir, lbl_dir = f"{ROOT}/images/{split}", f"{ROOT}/labels/{split}"
    tp = {c: [] for c in range(NC)}; fp = {c: [] for c in range(NC)}; ngt = {c: 0 for c in range(NC)}
    for r in m.predict(source=img_dir, conf=0.001, iou=0.6, stream=True, verbose=False, device=2):
        H, W = r.orig_shape
        stem = os.path.splitext(os.path.basename(r.path))[0]
        lbl = os.path.join(lbl_dir, stem + ".txt")
        gts = []
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5:
                    c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                    gts.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
        for gc, _ in gts:
            ngt[gc] += 1
        cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); bx = r.boxes.xyxy.cpu().numpy()
        used = set()
        for k in np.argsort(-cf):
            pc = int(cl[k]); best, bj = IOU_T, -1
            for j, (gc, gb) in enumerate(gts):
                if j in used or gc != pc:
                    continue
                v = iou(bx[k], gb)
                if v >= best:
                    best, bj = v, j
            (tp[pc].append(float(cf[k])), used.add(bj)) if bj >= 0 else fp[pc].append(float(cf[k]))
    return tp, fp, ngt


def rp(tp, fp, ngt, c, T):
    tpc = sum(x >= T for x in tp[c]); fpc = sum(x >= T for x in fp[c])
    rec = tpc / ngt[c] if ngt[c] else 0.0
    prec = tpc / (tpc + fpc) if (tpc + fpc) else 0.0
    return rec, prec


tp_tr, fp_tr, ngt_tr = collect("train")
thr = {}
for c in range(NC):
    if ngt_tr[c] == 0:
        continue
    bestf2, bestT = -1, 0.25
    for T in np.arange(0.03, 0.50, 0.01):
        rec, prec = rp(tp_tr, fp_tr, ngt_tr, c, T)
        f2 = (5*prec*rec)/(4*prec+rec) if (4*prec+rec) > 0 else 0
        if f2 > bestf2:
            bestf2, bestT = f2, round(float(T), 2)
    thr[c] = bestT

tp_v, fp_v, ngt_v = collect("val")
print("Dataset B (val)  baseline conf=0.25  vs  per-class 自适应阈值")
print("  cls  GT   T*   R_base->R_adp    P_base->P_adp")
recs_b, recs_a, precs_b, precs_a = [], [], [], []
for c in range(NC):
    if ngt_v[c] == 0:
        continue
    rb, pb = rp(tp_v, fp_v, ngt_v, c, 0.25)
    ra, pa = rp(tp_v, fp_v, ngt_v, c, thr[c])
    recs_b.append(rb); recs_a.append(ra); precs_b.append(pb); precs_a.append(pa)
    flag = "  <-- 难类" if rb < 0.35 else ""
    print(f"  {c:<3} {ngt_v[c]:>4}  {thr[c]:.2f}  {rb:.2f}->{ra:.2f}     {pb:.2f}->{pa:.2f}{flag}")

print(f"\n  macro Recall:    {np.mean(recs_b):.3f} -> {np.mean(recs_a):.3f}   (+{np.mean(recs_a)-np.mean(recs_b):.3f})")
print(f"  macro Precision: {np.mean(precs_b):.3f} -> {np.mean(precs_a):.3f}   ({np.mean(precs_a)-np.mean(precs_b):+.3f})")
hb = [recs_b[i] for i in range(len(recs_b)) if recs_b[i] < 0.35]
ha = [recs_a[i] for i in range(len(recs_b)) if recs_b[i] < 0.35]
print(f"  难类 macro Recall: {np.mean(hb):.3f} -> {np.mean(ha):.3f}   (+{np.mean(ha)-np.mean(hb):.3f})")
