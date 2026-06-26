#!/usr/bin/env python
import json, sys
RES = "/home/fenghn/CvTest/experiments/results"
TOTAL_B = 7728
for name in ("yolov8m_B", "yolo11m_B"):
    d = json.load(open(f"{RES}/{name}.json"))
    pc = d["per_class"]
    rows = sorted(pc.items(), key=lambda kv: -kv[1]["n_inst"])
    print(f"\n== {name}  (per-class, 按实例数降序) ==")
    print("  cls  name         n_inst   Recall   AP50   tier")
    for k, v in rows:
        n = v["n_inst"]; nm = str(v["name"])[:10]; r = v["recall"]; ap = v["ap50"]
        tier = "head" if n >= 0.10 * TOTAL_B else "tail" if n <= 0.02 * TOTAL_B else "med"
        print(f"  {int(k):<3}  {nm:<10} {n:>6}   {r:.3f}   {ap:.3f}  [{tier}]")
