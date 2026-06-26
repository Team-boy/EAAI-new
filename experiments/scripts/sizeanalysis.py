#!/usr/bin/env python
import json, glob, statistics
from collections import defaultdict

RES = "/home/fenghn/CvTest/experiments/results"
ROOT = "/home/fenghn/CvTest/data/Dataset B"

areas = defaultdict(list)
for d in glob.glob(ROOT + "/labels/*"):
    for f in glob.glob(d + "/*.txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0])); w = float(p[3]); h = float(p[4])
                areas[c].append(w * h)

d = json.load(open(RES + "/yolov8m_B.json"))
pc = d["per_class"]
rows = []
for k, v in pc.items():
    c = int(k); ar = areas.get(c, [])
    med = statistics.median(ar) if ar else 0.0
    rows.append((v["recall"], c, str(v["name"]), v["n_inst"], med))
rows.sort()   # 按 recall 升序: 最难在最上

print("按 Recall 升序 (最难在上)  —  medArea% = bbox 面积中位数占整图%")
print("  cls  name        n_inst   medArea%   Recall")
for r in rows:
    print(f"  {r[1]:<3}  {r[2][:8]:<8} {r[3]:>6}    {r[4]*100:7.3f}    {r[0]:.3f}")

# 相关性: recall vs size, recall vs freq
import math
def corr(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx*sy else 0.0

rec = [r[0] for r in rows]; siz = [r[4] for r in rows]; frq = [r[3] for r in rows]
print(f"\n  corr(Recall, size)      = {corr(rec, siz):+.3f}   <- 越接近+1 说明'越大越好检/越小越难'")
print(f"  corr(Recall, frequency) = {corr(rec, frq):+.3f}   <- 接近0 说明频次不predict难度")
