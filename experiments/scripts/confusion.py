#!/usr/bin/env python
import statistics
from ultralytics import YOLO

m = YOLO("/home/fenghn/CvTest/experiments/runs/yolov8m_B/weights/best.pt")
res = m.val(data="/home/fenghn/CvTest/data/Dataset B/data.yaml",
            split="val", device=2, verbose=False, plots=False)
cm = res.confusion_matrix.matrix          # matrix[pred, gt], 最后一行/列=背景
names = res.names
if isinstance(names, dict):
    names = [names[i] for i in range(len(names))]
nc = cm.shape[0] - 1
print(f"DEBUG cm.shape={cm.shape} sum={float(cm.sum()):.1f} diag5={[int(cm[i,i]) for i in range(min(5,cm.shape[0]))]}")
print(f"DEBUG col_sums(前6)={[int(cm[:,j].sum()) for j in range(6)]}")

rows = []
for j in range(nc):                        # 列 j = 真值类 j
    col = cm[:, j]; tot = col.sum()
    if tot == 0:
        continue
    correct = cm[j, j] / tot               # 检对
    bg = cm[nc, j] / tot                    # 漏到背景 (FN)
    conf = sorted(((cm[i, j], i) for i in range(nc) if i != j), reverse=True)
    top_v, top_i = conf[0]
    rows.append((correct, j, int(tot), bg, top_i, top_v / tot))

rows.sort()                                # 正确率升序: 最难在上
print("Dataset B 混淆分解 (按正确率升序, 最难在上)")
print("  cls   GT    正确   漏到背景   最大混淆类(占比)")
for c, j, tot, bg, ti, tv in rows:
    print(f"  {j:<3}  {tot:>4}   {c:.2f}     {bg:.2f}      cls{ti}({tv:.2f})")

hard = [r for r in rows if r[0] < 0.35]
print(f"\n难类(正确率<0.35)共 {len(hard)} 个:")
if hard:
    print(f"  平均 漏到背景 = {statistics.mean(r[3] for r in hard):.2f}")
    print(f"  平均 最大单类混淆 = {statistics.mean(r[5] for r in hard):.2f}")
print("  >> 若'漏到背景' >> '混淆' => 纯漏检(走B); 若'混淆'可观 => 类间混淆(走A)")
