#!/usr/bin/env python
"""(a) miss-aware loss: 按 baseline per-class 漏检率给 BCE 设 pos_weight, 等预算重训."""
import argparse, json, torch, torch.nn as nn
from ultralytics.utils import loss as ul

p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=3.0)
p.add_argument("--data", required=True)
p.add_argument("--baseline_json", required=True)
p.add_argument("--nc", type=int, required=True)
p.add_argument("--device", default="0")
p.add_argument("--name", required=True)
p.add_argument("--epochs", type=int, default=200)
a = p.parse_args()

# 权重 = 1 + alpha*(1-recall_baseline); 空类保持 1; 归一化使均值=1 (保持总损失尺度)
rec = {int(k): v["recall"] for k, v in json.load(open(a.baseline_json))["per_class"].items()}
w = torch.ones(a.nc)
for c in range(a.nc):
    if c in rec:
        w[c] = 1.0 + a.alpha * (1.0 - rec[c])
w = w / w.mean()
print("miss-aware pos_weight (按类):", [round(float(x), 2) for x in w])
POS = w

_orig = ul.v8DetectionLoss.__init__
_applied = {"done": False}
def patched(self, *args, **kwargs):
    _orig(self, *args, **kwargs)
    dev = next(args[0].parameters()).device
    self.bce = nn.BCEWithLogitsLoss(pos_weight=POS.to(dev), reduction="none")
    if not _applied["done"]:
        print(f"[PATCH] v8DetectionLoss BCE pos_weight 已注入 (device={dev})")
        _applied["done"] = True
ul.v8DetectionLoss.__init__ = patched

from ultralytics import YOLO
m = YOLO("yolov8m.pt")
m.train(data=a.data, epochs=a.epochs, imgsz=640, batch=16, device=a.device, seed=0,
        project="/home/fenghn/CvTest/experiments/runs", name=a.name,
        exist_ok=True, verbose=False, plots=True, workers=8)
print("DONE", a.name)
