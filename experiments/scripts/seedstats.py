#!/usr/bin/env python
"""多-seed mean±std 聚合."""
import json, math
import numpy as np
RES = "/home/fenghn/CvTest/experiments/results"


def load(names):
    M = {"mAP50": [], "mAP50_95": [], "R_head": [], "R_med": [], "R_tail": []}
    for n in names:
        d = json.load(open(f"{RES}/{n}.json")); o = d["overall"]; g = d["groups"]
        M["mAP50"].append(o["mAP50"]); M["mAP50_95"].append(o["mAP50_95"])
        M["R_head"].append(g["head"]["recall"] if g["head"]["n_cls"] else math.nan)
        M["R_med"].append(g["med"]["recall"] if g["med"]["n_cls"] else math.nan)
        M["R_tail"].append(g["tail"]["recall"] if g["tail"]["n_cls"] else math.nan)
    return M


def show(M, tag):
    print(tag)
    for k, v in M.items():
        vv = [x for x in v if not math.isnan(x)]
        if vv:
            print(f"  {k:10}: {np.mean(vv):.3f} ± {np.std(vv):.3f}   (n={len(vv)}, seeds)")


show(load(["yolov8m_B", "yolov8m_B_s1", "yolov8m_B_s2"]), "Dataset B  YOLOv8m  (3 seeds)")
print()
show(load(["yolov8m_A", "yolov8m_A_s1", "yolov8m_A_s2"]), "Dataset A  YOLOv8m  (3 seeds)")
