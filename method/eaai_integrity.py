# -*- coding: utf-8 -*-
"""EAAI论文诚信复核: 核验论文数字 vs 源JSON/缓存."""
import json, os, pickle
import numpy as np

RES = "/home/fenghn/CvTest/experiments/results"
CACHE = "/home/fenghn/CvTest/method/vbad_cache"
IOU_T = 0.5


def ov(name):
    p = f"{RES}/{name}_TEST.json"
    return json.load(open(p))["overall"]["mAP50"] if os.path.exists(p) else None


def grp(name, g):
    p = f"{RES}/{name}_TEST.json"
    return json.load(open(p))["groups"][g]["recall"] if os.path.exists(p) else None


print("=== 表2/3 数字核验 (vs source JSON) ===")
checks = [
    ("640 baseline mAP", ov("yolov8m_B"), 0.397),
    ("960 mAP", ov("resB_v8m_960"), 0.496),
    ("1280 mAP", ov("opt1_hires1280"), 0.482),
    ("1536 mAP", ov("resB_v8m_1536"), 0.499),
    ("YOLO11m@1280", ov("resB_v11m_1280"), 0.515),
    ("miss-aware", ov("missaware_B"), 0.398),
    ("yolov8s", ov("opt2_yolov8s"), 0.365),
    ("P2", ov("opt3_p2"), 0.365),
    ("distill", ov("distill_640_kd"), 0.421),
    ("R_head@640", grp("yolov8m_B", "head"), 0.554),
    ("R_med@640", grp("yolov8m_B", "med"), 0.367),
    ("R_tail@640", grp("yolov8m_B", "tail"), 0.475),
]
ok = bad = 0
for name, actual, paper in checks:
    if actual is None:
        print(f"  ? {name}: source missing"); continue
    match = abs(actual - paper) < 0.002
    print(f"  {'OK ' if match else 'BAD'} {name}: source={actual:.3f} paper={paper}")
    ok += match; bad += (not match)

s = [ov("opt1_hires1280"), ov("resB_v8m_1280_s1"), ov("resB_v8m_1280_s2")]
m, sd = np.mean(s), np.std(s)
m_ok = abs(m-0.501) < 0.002 and abs(sd-0.014) < 0.003
print(f"  {'OK ' if m_ok else 'BAD'} 1280 3-seed: source={m:.3f}±{sd:.3f} paper=0.501±.014")
ok += m_ok; bad += (not m_ok)

print("\n=== 表4/5 + VBAD Type-V/A (vs cache) ===")
te = pickle.load(open(f"{CACHE}/test2.pkl", "rb"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def hit(g, ps): return any(p[0] == g[0] and iou(g[1:5], p[1:5]) >= IOU_T for p in ps)


import collections
cls = collections.defaultdict(lambda: [0, 0, 0, 0])
tot = h640 = h1280 = tv = ta = 0
for r in te:
    for g in r["gts"]:
        tot += 1
        a = hit(g, r["d640"]); b = hit(g, r["d1280"])
        h640 += a; h1280 += b; tv += (not a) and b; ta += (not b)
        cls[g[0]][0] += 1; cls[g[0]][1] += a; cls[g[0]][2] += b; cls[g[0]][3] += (not b)
print(f"  total GT {tot} (paper 1625): {'OK' if tot==1625 else 'BAD'}")
print(f"  R@640={h640/tot:.3f}(paper .410) R@1280={h1280/tot:.3f}(paper .572)")
print(f"  Type-V {tv/tot:.1%}(paper 21.5%) Type-A {ta/tot:.1%}(paper 42.8%)")
for c, pa in [(7, 73), (10, 65), (11, 44), (13, 74), (17, 34)]:
    t = cls[c]
    print(f"    cls{c}: R640={t[1]/t[0]:.2f} R1280={t[2]/t[0]:.2f} TypeA={t[3]/t[0]*100:.0f}% (paper {pa}%)")
print(f"\n表2/3核验: {ok} OK, {bad} BAD")
