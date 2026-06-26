#!/usr/bin/env python
"""汇总 results/*.json 为长尾对比表 (markdown)."""
import json, glob, sys, os

RES = "/home/fenghn/CvTest/experiments/results"
names = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(RES + "/*.json"))

rows = []
for n in names:
    p = os.path.join(RES, n + ".json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p)); o = d["overall"]; g = d["groups"]
    cell = lambda grp, k: (f"{grp[k]:.3f}" if grp["n_cls"] > 0 else "-")
    rows.append([d["meta"]["data"], d["meta"]["model"],
                 f"{o['mAP50']:.3f}", f"{o['mAP50_95']:.3f}",
                 cell(g["head"], "recall"), cell(g["med"], "recall"), cell(g["tail"], "recall"),
                 cell(g["head"], "mAP50"), cell(g["med"], "mAP50"), cell(g["tail"], "mAP50")])

rows.sort()
hdr = ["dataset", "model", "mAP50", "mAP50-95", "R_head", "R_med", "R_tail", "AP_head", "AP_med", "AP_tail"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r in rows:
    print("| " + " | ".join(r) + " |")
print(f"\n({len(rows)} runs aggregated)  R=Recall, AP=mAP50; head>=10% / tail<=2% instances; '-'=该组无类")
