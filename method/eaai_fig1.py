# -*- coding: utf-8 -*-
"""EAAI fig1: VBAD framework 流程图 (matplotlib 画框图)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

plt.rcParams.update({"font.family": "serif", "font.size": 9})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "grey": "#CCCCCC", "lgrey": "#EEEEEE"}

fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")


def box(x, y, w, h, text, fc, ec="black", fs=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(x1, y1, x2, y2, text="", color="black"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color=color))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.25, text, ha="center", fontsize=7.5, color=color)


# 输入
box(0.3, 3.2, 1.8, 1.4, "Input\nimage", CB["lgrey"], fs=9)
# 640 detector
box(2.8, 3.2, 2.0, 1.4, "Low-res\ndetector\n(640)", CB["blue"], fs=8.5, tc="white")
arrow(2.1, 3.9, 2.8, 3.9)
# 输出: 检测 + 特征
box(5.5, 5.2, 2.2, 1.2, "Detections\n$D_{640}$", CB["lgrey"], fs=8.5)
box(5.5, 1.4, 2.2, 1.2, "Feature\ndescriptor $\\phi$", CB["lgrey"], fs=8.5)
arrow(4.8, 4.2, 5.5, 5.6)
arrow(4.8, 3.6, 5.5, 2.2)
# visibility head
box(8.3, 1.3, 2.2, 1.4, "Visibility\nhead", CB["green"], fs=8.5, tc="white")
arrow(7.7, 2.0, 8.3, 2.0)
# 决策菱形(用box近似)
ax.text(11.0, 2.0, "$s_{vis} \\geq \\tau$?", ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec=CB["green"]))
arrow(10.5, 2.0, 10.3, 2.0)
# 高分辨率复检 (触发)
box(11.6, 4.6, 2.2, 1.4, "Full-image\nhigh-res (1280)\nreinspect", CB["red"], fs=8, tc="white")
arrow(11.0, 2.7, 12.0, 4.6, "yes", CB["red"])
# 保持640 (不触发)
arrow(11.6, 1.5, 10.2, 0.6, "no", CB["grey"])
# 合并/选择输出
box(8.0, 5.2, 2.0, 1.2, "Select\nfinal\ndetections", CB["orange"], fs=8.5)
arrow(11.6, 5.3, 10.0, 5.8, "", CB["red"])      # 1280->select
arrow(7.7, 5.8, 8.0, 5.8, "", CB["grey"])        # 640 dets->select
# 监督信号注释
ax.text(9.4, 0.35, "Type-V label = (640 miss) $\\wedge$ (1280 hit), cross-resolution failure supervision",
        ha="center", fontsize=7, style="italic", color=CB["green"])

fig.tight_layout()
fig.savefig("/home/fenghn/CvTest/paper_eaai/figs/fig1_framework.pdf", bbox_inches="tight")
fig.savefig("/home/fenghn/CvTest/paper_eaai/figs/fig1_framework.png", dpi=160, bbox_inches="tight")
print("fig1 framework saved")
