# -*- coding: utf-8 -*-
"""EAAI fig4: Type-V / Type-A 真实缺陷样例可视化.
每行一个样例: 原图裁剪(GT区域) + GT框(绿) + 640检测(蓝) + 1280检测(红).
Type-V行: 640漏1280救; Type-A行: 1280也漏."""
import os, glob, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
CACHE = "/home/fenghn/CvTest/method/vbad_cache"
FIG = "/home/fenghn/CvTest/paper_eaai/figs"
IOU_T = 0.5
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def hit(g, ps):
    return any(p[0] == g[0] and iou(g[1:5], p[1:5]) >= IOU_T for p in ps)


te = pickle.load(open(f"{CACHE}/test2.pkl", "rb"))
# 把 stem -> 图路径
img_map = {}
for p in glob.glob(f"{ROOT}/images/test2/*"):
    img_map[os.path.splitext(os.path.basename(p))[0]] = p

# 找典型样例: Type-V (640漏,1280救), Type-A (1280也漏), 优先小目标且单缺陷清晰
typev, typea = [], []
for r in te:
    stem = r["stem"]
    if stem not in img_map:
        continue
    for g in r["gts"]:
        a = hit(g, r["d640"]); b = hit(g, r["d1280"])
        area = g[3]*g[4]
        if (not a) and b and 0.0002 < area < 0.02:           # Type-V, 小而非极小
            typev.append((stem, g, r))
        elif (not b) and area < 0.01:                          # Type-A, 小
            typea.append((stem, g, r))

# 各取3个
sel_v = typev[:3]; sel_a = typea[:3]
rows = [("Type-V (recovered at 1280)", sel_v), ("Type-A (unrecoverable)", sel_a)]

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))
for ri, (title, sel) in enumerate(rows):
    for ci in range(3):
        ax = axes[ri][ci]
        ax.axis("off")
        if ci >= len(sel):
            continue
        stem, g, r = sel[ci]
        im = Image.open(img_map[stem]).convert("RGB")
        W, H = im.size
        # 裁 GT 周围 patch (放大显示)
        cx, cy, gw, gh = g[1]*W, g[2]*H, g[3]*W, g[4]*H
        side = max(gw, gh) * 6 + 40
        x1, y1 = max(0, int(cx-side/2)), max(0, int(cy-side/2))
        x2, y2 = min(W, int(cx+side/2)), min(H, int(cy+side/2))
        crop = im.crop((x1, y1, x2, y2)); cw, ch = crop.size
        ax.imshow(crop)
        # GT 框(绿) — 转到 crop 坐标
        gx1 = (cx-gw/2-x1); gy1 = (cy-gh/2-y1)
        ax.add_patch(Rectangle((gx1, gy1), gw, gh, fill=False, ec="#00CC00", lw=1.8))
        if ri == 0:  # Type-V: 画 1280 检出框(红虚线)
            for p in r["d1280"]:
                if p[0] == g[0] and iou(g[1:5], p[1:5]) >= IOU_T:
                    px1 = (p[1]*W-p[3]*W/2-x1); py1 = (p[2]*H-p[4]*H/2-y1)
                    ax.add_patch(Rectangle((px1, py1), p[3]*W, p[4]*H, fill=False, ec="#D55E00", lw=1.3, ls="--"))
        ax.set_title(f"GT cls{g[0]}, area {g[3]*g[4]*100:.2f}%", fontsize=7)
    axes[ri][0].set_ylabel(title, fontsize=9)
    axes[ri][0].axis("on"); axes[ri][0].set_xticks([]); axes[ri][0].set_yticks([])
    for s in axes[ri][0].spines.values():
        s.set_visible(False)

# 图例
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color="#00CC00", lw=2, label="Ground truth"),
       Line2D([0], [0], color="#D55E00", lw=1.5, ls="--", label="1280 detection")]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.suptitle("Type-V vs Type-A defect examples", fontsize=10)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(f"{FIG}/fig4_examples.pdf"); fig.savefig(f"{FIG}/fig4_examples.png", dpi=160)
print(f"fig4 saved. Type-V候选{len(typev)}, Type-A候选{len(typea)}, 用了 V{len(sel_v)} A{len(sel_a)}")
