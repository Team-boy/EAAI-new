# -*- coding: utf-8 -*-
"""Stage 4.5: 核验改后论文数字 vs 源 JSON (test 集)."""
import json
import statistics as st
RES = "/home/fenghn/CvTest/experiments/results"

def g(n):
    d = json.load(open(f"{RES}/{n}.json")); o = d["overall"]; gr = d["groups"]
    return {"mAP50": o["mAP50"], "mAP50_95": o["mAP50_95"],
            "head": gr["head"]["recall"], "med": gr["med"]["recall"], "tail": gr["tail"]["recall"]}

print("=== Table 1 / §4 核验 (test) ===")
# A YOLOv8m
a = g("yolov8m_A_TEST")
print(f"A YOLOv8m: mAP50={a['mAP50']:.3f}(纸0.906) mAP5095={a['mAP50_95']:.3f}(纸0.548) head={a['head']:.3f}(纸0.877)")
# B 3-seed clean (ts, ts_s1, ts_s2)
seeds = ["yolov8m_B_TEST", "yolov8m_B_ts_s1_TEST", "yolov8m_B_ts_s2_TEST"]
M = [g(s) for s in seeds]
for k, paper in [("mAP50", "0.387±.010"), ("mAP50_95", "0.187±.006"),
                 ("head", "0.548±.018"), ("med", "0.356±.013"), ("tail", "0.468±.013")]:
    vals = [m[k] for m in M]
    print(f"B 3seed {k:9}: {st.mean(vals):.3f}±{st.pstdev(vals):.3f}  (纸 {paper})")
# YOLOv11m / RT-DETR single
for run, lab in [("yolo11m_B_TEST", "YOLOv11m 纸 .385/.584/.336/.500"),
                 ("rtdetr_B_TEST", "RT-DETR 纸 .363/.321/.366/.520")]:
    m = g(run)
    print(f"B {run}: mAP50={m['mAP50']:.3f} head={m['head']:.3f} med={m['med']:.3f} tail={m['tail']:.3f}  ({lab})")

print("\n=== Table 2 (method, test) 核验 ===")
b = g("yolov8m_B_TEST"); ma = g("missaware_B_TEST")
print(f"baseline: mAP50={b['mAP50']:.3f}(纸0.397) head={b['head']:.3f}(纸0.554) med={b['med']:.3f}(纸0.367) tail={b['tail']:.3f}(纸0.475)")
print(f"missaware:mAP50={ma['mAP50']:.3f}(纸0.398) head={ma['head']:.3f}(纸0.590) med={ma['med']:.3f}(纸0.381) tail={ma['tail']:.3f}(纸0.495)")

print("\n=== RT-DETR head<tail (§4.2 R_tail=0.52>R_head=0.32) 核验 ===")
r = g("rtdetr_B_TEST")
print(f"  R_tail={r['tail']:.2f} R_head={r['head']:.2f}  -> tail>head? {r['tail']>r['head']}")
