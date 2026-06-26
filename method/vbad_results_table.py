# -*- coding: utf-8 -*-
"""VBAD论文 表2/3/4/5 数据汇总: 从现有JSON + VBAD缓存一次性生成所有数字."""
import json, os, pickle
import numpy as np

RES = "/home/fenghn/CvTest/experiments/results"
CACHE = "/home/fenghn/CvTest/method/vbad_cache"   # v1 GAP特征缓存(含检测+TypeV)
IOU_T = 0.5


def load(name):
    p = f"{RES}/{name}_TEST.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def ov(name):
    d = load(name)
    return d["overall"]["mAP50"] if d else None


# ===== 表3: 主结果(分辨率方法 on B test2) =====
print("="*60)
print("表3 主结果 (Dataset B, test2)")
print("="*60)
print(f"{'method':24} {'mAP50':>7} {'R_head':>7} {'R_med':>7} {'R_tail':>7}")
for nm, key in [("640 baseline", "yolov8m_B"), ("960", "resB_v8m_960"),
                ("1280 (full HR)", "opt1_hires1280"), ("1536", "resB_v8m_1536"),
                ("YOLO11m@1280", "resB_v11m_1280")]:
    d = load(key)
    if d:
        o, g = d["overall"], d["groups"]
        print(f"{nm:24} {o['mAP50']:7.3f} {g['head']['recall']:7.3f} {g['med']['recall']:7.3f} {g['tail']['recall']:7.3f}")

# 1280 三seed mean
s = [ov("opt1_hires1280"), ov("resB_v8m_1280_s1"), ov("resB_v8m_1280_s2")]
print(f"  1280 3-seed mAP50: {np.mean(s):.3f}±{np.std(s):.3f}")

# ===== 表2: 失败方法总结(on B test2) =====
print("\n" + "="*60)
print("表2 失败/对照方法总结 (Dataset B, test2 mAP50)")
print("="*60)
for nm, key in [("640 baseline", "yolov8m_B"), ("miss-aware loss", "missaware_B"),
                ("yolov8s(anti-overfit)", "opt2_yolov8s"), ("P2 small-obj head", "opt3_p2"),
                ("distillation 1280->640", "distill_640_kd"), ("full 1280", "opt1_hires1280")]:
    v = ov(key)
    print(f"  {nm:26} {v:.3f}" if v else f"  {nm:26} N/A")
print("  (密度分支: cls18 recall 0.03 FAIL; VGR: oracle 0.463<1280; 自适应: 触发器够不着oracle)")

# ===== 表4 + 表5: VBAD分诊 + Type-A 分析(从缓存算) =====
print("\n" + "="*60)
print("表4/5 VBAD分诊 + Type-A 不可恢复分析 (test2, GT级)")
print("="*60)
te = pickle.load(open(f"{CACHE}/test2.pkl", "rb"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def hit(gt, preds):
    return any(p[0] == gt[0] and iou(gt[1:5], p[1:5]) >= IOU_T for p in preds)


import collections
cls = collections.defaultdict(lambda: [0, 0, 0, 0])  # cls -> [tot, hit640, hit1280, typeA]
tot = h640 = h1280 = typeV = typeA = 0
for r in te:
    for g in r["gts"]:
        tot += 1
        a = hit(g, r["d640"]); b = hit(g, r["d1280"])
        h640 += a; h1280 += b
        if (not a) and b:
            typeV += 1
        if not b:
            typeA += 1
        cls[g[0]][0] += 1; cls[g[0]][1] += a; cls[g[0]][2] += b; cls[g[0]][3] += (not b)
print(f"  total GT {tot}: R@640={h640/tot:.3f}, R@1280={h1280/tot:.3f}")
print(f"  Type-V (640miss,1280hit)= {typeV} ({typeV/tot:.1%})  <- VBAD可救")
print(f"  Type-A (1280also miss)  = {typeA} ({typeA/tot:.1%})  <- 不可恢复天花板")
print("\n  表4 极小目标类 (640->1280 recall, Type-A率):")
SMALL = {7: "nep", 17: "spandex-break", 10: "broken-warp", 13: "weft-shrink", 11: "hanging-warp"}
for c in sorted(SMALL):
    if c in cls:
        t, a, b, ta = cls[c]
        print(f"    cls{c}({SMALL[c]}): R640={a/t:.2f} R1280={b/t:.2f} Type-A={ta/t:.0%} (n={t})")
print("\nDONE")
