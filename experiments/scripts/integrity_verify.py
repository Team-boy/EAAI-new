#!/usr/bin/env python
"""Stage 2.5 数值核验: 论文数字 vs 源 JSON; 相关性显著性; round-number 红旗."""
import json, math, glob, statistics, collections
import numpy as np
RES = "/home/fenghn/CvTest/experiments/results"
ROOT = "/home/fenghn/CvTest/data/Dataset B"

# --- Mode 3: 论文 headline 数字是否 = 源 JSON ---
def grp(n):
    d = json.load(open(f"{RES}/{n}.json")); return d["overall"], d["groups"]
seeds = ["yolov8m_B", "yolov8m_B_s1", "yolov8m_B_s2"]
mAP = [grp(s)[0]["mAP50"] for s in seeds]
rmed = [grp(s)[1]["med"]["recall"] for s in seeds]
print("[Mode3 数值核验] B YOLOv8m 3-seed (源 JSON):")
print(f"  mAP50  = {np.mean(mAP):.3f} ± {np.std(mAP):.3f}   (论文写 0.420±0.004)")
print(f"  R_med  = {np.mean(rmed):.3f} ± {np.std(rmed):.3f}   (论文写 0.394±0.007)")

# --- 相关性 + 近似 p 值 (Mode 1/统计严谨) ---
pc = json.load(open(f"{RES}/yolov8m_B.json"))["per_class"]
rec = {int(k): v["recall"] for k, v in pc.items()}; freq = {int(k): v["n_inst"] for k, v in pc.items()}
areas = collections.defaultdict(list)
for dd in glob.glob(ROOT+"/labels/*"):
    for f in glob.glob(dd+"/*.txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 5: areas[int(float(p[0]))].append(float(p[3])*float(p[4]))
cls = sorted(rec)
size = {c: (statistics.median(areas[c]) if areas[c] else 0) for c in cls}
def corr(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(sx*sy) if sx*sy else 0
def pval(r, n):
    if abs(r) >= 1: return 0.0
    t = r*math.sqrt((n-2)/(1-r*r))
    # 双尾 p 近似 (正态近似, n=20 够用作显著性判断)
    z = abs(t); p = math.erfc(z/math.sqrt(2))
    return p
ys = [rec[c] for c in cls]; n = len(cls)
for name, xs in [("freq", [freq[c] for c in cls]), ("size", [size[c] for c in cls])]:
    r = corr(xs, ys); p = pval(r, n)
    print(f"[统计] recall~{name}: r={r:+.3f}, n={n}, p≈{p:.2f}  -> {'不显著(支持无相关)' if p>0.05 else '显著'}")

# --- Mode 1 红旗: 可疑整数 / 相同误差棒 ---
print("[Mode1 红旗扫描]")
print(f"  R_tail std = {np.std([grp(s)[1]['tail']['recall'] for s in seeds]):.3f} (大方差→tail组仅3类,合理非bug)")
print(f"  mAP50 3seed 是否完全相同? {'是(可疑!)' if len(set([round(x,4) for x in mAP]))==1 else '否(各seed有差异,正常)'}")
