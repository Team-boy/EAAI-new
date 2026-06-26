#!/usr/bin/env python
"""test2 上的完整诊断: 相关性p值 + 跨检测器漏检/混淆/余量 + 3seed mean±std."""
import json, math, glob, statistics, collections, os
import numpy as np
from ultralytics import YOLO, RTDETR

RES = "/home/fenghn/CvTest/experiments/results"
ROOT = "/home/fenghn/CvTest/data/Dataset B"
RUNS = "/home/fenghn/CvTest/experiments/runs"
NC, IOU_T = 21, 0.5

# ---- 相关性 p 值 (test) ----
pc = json.load(open(f"{RES}/yolov8m_B_TEST.json"))["per_class"]
rec = {int(k): v["recall"] for k, v in pc.items()}
freq = {int(k): v["n_inst"] for k, v in pc.items()}
ar = collections.defaultdict(list)
for dd in glob.glob(ROOT + "/labels/train2/*.txt"):
    for line in open(dd):
        s = line.split()
        if len(s) >= 5:
            ar[int(float(s[0]))].append(float(s[3]) * float(s[4]))
cls = sorted(rec)
size = {c: (statistics.median(ar[c]) if ar[c] else 0) for c in cls}
def corr(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(sx*sy) if sx*sy else 0
def pv(r, n):
    t = r*math.sqrt((n-2)/(1-r*r)); return math.erfc(abs(t)/math.sqrt(2))
ys = [rec[c] for c in cls]; n = len(cls)
print("== 相关性 [TEST] ==")
for nm, xs in [("freq", [freq[c] for c in cls]), ("size", [size[c] for c in cls])]:
    r = corr(xs, ys); p = pv(r, n)
    print(f"  recall~{nm}: r={r:+.3f} n={n} p={p:.2f} {'n.s.' if p > 0.05 else 'SIG'}")

# ---- 难类(按 test R<0.35 重定) + 跨检测器漏检/混淆/余量 ----
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1)*max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0
hard = sorted([c for c in cls if rec[c] < 0.35])
print(f"\n== 难类(test R<0.35): {hard} ==")
def analyze(weights):
    Model = RTDETR if "rtdetr" in weights.lower() else YOLO
    m = Model(weights)
    same = {c: [] for c in range(NC)}; dec = {c: [0, 0, 0] for c in range(NC)}
    for r in m.predict(source=ROOT+"/images/test2", conf=0.001, iou=0.6, stream=True, verbose=False, device=7):
        H, W = r.orig_shape
        stem = os.path.splitext(os.path.basename(r.path))[0]
        lbl = os.path.join(ROOT, "labels/test2", stem+".txt")
        gts = []
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5:
                    c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                    gts.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
        cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); bx = r.boxes.xyxy.cpu().numpy()
        for gc, gb in gts:
            sc, o25 = 0.0, False
            for k in range(len(cf)):
                if iou(bx[k], gb) >= IOU_T:
                    if cl[k] == gc: sc = max(sc, float(cf[k]))
                    elif cf[k] >= 0.25: o25 = True
            same[gc].append(sc); dec[gc][0 if sc >= 0.25 else (1 if o25 else 2)] += 1
    miss = np.mean([dec[c][2]/sum(dec[c]) for c in hard if sum(dec[c])])
    conf = np.mean([dec[c][1]/sum(dec[c]) for c in hard if sum(dec[c])])
    r25 = np.mean([np.mean(np.array(same[c]) >= 0.25) for c in hard if same[c]])
    rmx = np.mean([np.mean(np.array(same[c]) > 0.0) for c in hard if same[c]])
    return miss, conf, r25, rmx
print("  detector   missed confused R@.25 R@max")
for name, run in [("YOLOv8m", "yolov8m_B_ts"), ("YOLOv11m", "yolo11m_B_ts"), ("RT-DETR", "rtdetr_B_ts")]:
    mi, co, r25, rmx = analyze(f"{RUNS}/{run}/weights/best.pt")
    print(f"  {name:9} {mi:.2f}   {co:.2f}    {r25:.2f}  {rmx:.2f}")

# ---- 3-seed test mean±std ----
def gg(nm):
    d = json.load(open(f"{RES}/{nm}.json"))["groups"]; o = json.load(open(f"{RES}/{nm}.json"))["overall"]
    return o["mAP50"], d["head"]["recall"], d["med"]["recall"], d["tail"]["recall"]
seeds = ["yolov8m_B_TEST", "yolov8m_B_s1_TEST", "yolov8m_B_s2_TEST"]
arr = np.array([gg(s) for s in seeds])
print("\n== 3-seed [TEST] mean±std (s1/s2 略大训练池, 稳健性参考) ==")
for i, lab in enumerate(["mAP50", "R_head", "R_med", "R_tail"]):
    print(f"  {lab:7} {arr[:,i].mean():.3f} ± {arr[:,i].std():.3f}")
