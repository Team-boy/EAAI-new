#!/usr/bin/env python
"""在冻结 test2 上评测重训模型, 输出 *_ts.json (复用 run_exp 的分组逻辑, split=test)."""
import argparse, json, os, glob
from collections import Counter
import yaml

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--data", required=True)      # data_split.yaml (有 test 字段)
p.add_argument("--device", default="0")
p.add_argument("--name", required=True)
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()

from ultralytics import YOLO, RTDETR
Model = RTDETR if "rtdetr" in a.weights.lower() else YOLO

root = os.path.dirname(os.path.abspath(a.data))
cfg = yaml.safe_load(open(a.data))
names = cfg["names"]
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]
nc = len(names)
# 频次按 train2 统计 (分组依据应来自训练分布)
cnt = Counter()
for d in glob.glob(f"{root}/labels/train2/*.txt"):
    for line in open(d):
        s = line.strip()
        if s:
            cnt[int(float(s.split()[0]))] += 1
total = sum(cnt.values()) or 1

m = Model(a.weights)
res = m.val(data=a.data, split="test", imgsz=a.imgsz, device=a.device, verbose=False, plots=False)
box = res.box
idx = [int(c) for c in box.ap_class_index]
rec = {idx[i]: float(box.r[i]) for i in range(len(idx))}
ap50 = {idx[i]: float(box.ap50[i]) for i in range(len(idx))}
ap = {idx[i]: float(box.ap[i]) for i in range(len(idx))}

def groups():
    H, M, T = [], [], []
    for c in range(nc):
        n = cnt.get(c, 0)
        if n == 0:
            continue
        f = n/total
        (H if f >= .10 else T if f <= .02 else M).append(c)
    return H, M, T
H, M, Tt = groups()
agg = lambda g, d: (sum(d.get(c, 0) for c in g)/len(g) if g else 0.0)
out = {"overall": {"mAP50": float(box.map50), "mAP50_95": float(box.map),
                   "recall_mean": float(box.mr), "precision_mean": float(box.mp)},
       "groups": {g: {"classes": grp, "n_cls": len(grp), "recall": agg(grp, rec),
                      "mAP50": agg(grp, ap50), "mAP50_95": agg(grp, ap)}
                  for g, grp in (("head", H), ("med", M), ("tail", Tt))},
       "per_class": {c: {"name": str(names[c]), "n_inst": cnt.get(c, 0),
                         "recall": rec.get(c, 0.0), "ap50": ap50.get(c, 0.0)} for c in idx},
       "meta": {"weights": a.weights, "split": "test2", "name": a.name}}
os.makedirs("/home/fenghn/CvTest/experiments/results", exist_ok=True)
json.dump(out, open(f"/home/fenghn/CvTest/experiments/results/{a.name}.json", "w"), indent=2, ensure_ascii=False)
o, g = out["overall"], out["groups"]
print(f"=== {a.name} [TEST] === mAP50={o['mAP50']:.3f} mAP50-95={o['mAP50_95']:.3f}")
for k in ("head", "med", "tail"):
    gg = g[k]
    print(f"  {k:4}({gg['n_cls']:>2}): R={gg['recall']:.3f} AP50={gg['mAP50']:.3f}")
