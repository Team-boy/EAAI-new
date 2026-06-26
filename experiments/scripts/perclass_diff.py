#!/usr/bin/env python
"""对比 baseline vs miss-aware 的 per-class recall."""
import json, statistics
RES = "/home/fenghn/CvTest/experiments/results"
base = json.load(open(f"{RES}/yolov8m_B.json"))
new = json.load(open(f"{RES}/missaware_B.json"))
b, n = base["per_class"], new["per_class"]

rows = []
for k in b:
    rb = b[k]["recall"]; rn = n.get(k, {}).get("recall", 0.0); cnt = b[k]["n_inst"]
    rows.append((rb, int(k), cnt, rb, rn, rn - rb))
rows.sort()

print("per-class Recall: baseline -> miss-aware")
print("  cls   GT   R_base  R_miss    Δ")
for rb, c, cnt, a, bb, d in rows:
    flag = "  <-- 难类" if a < 0.35 else ""
    print(f"  {c:<3} {cnt:>5}   {a:.2f}    {bb:.2f}   {d:+.2f}{flag}")

hard = [r for r in rows if r[3] < 0.35]
print(f"\n  overall mAP50:     {base['overall']['mAP50']:.3f} -> {new['overall']['mAP50']:.3f}")
print(f"  overall mAP50-95:  {base['overall']['mAP50_95']:.3f} -> {new['overall']['mAP50_95']:.3f}")
print(f"  全类 macro Recall: {statistics.mean(r[3] for r in rows):.3f} -> {statistics.mean(r[4] for r in rows):.3f} "
      f"({statistics.mean(r[4]-r[3] for r in rows):+.3f})")
print(f"  难类 macro Recall: {statistics.mean(r[3] for r in hard):.3f} -> {statistics.mean(r[4] for r in hard):.3f} "
      f"({statistics.mean(r[4]-r[3] for r in hard):+.3f})")
print("  >> miss-aware 训练若在 同等mAP/精度 下抬升难类recall = PR曲线上移(成功); 若 mAP 掉 = 只是trade-off")
