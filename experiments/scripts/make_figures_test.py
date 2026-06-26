#!/usr/bin/env python
"""生成诊断驱动论文的 Fig1-3 + Table1-2(LaTeX). 数据来自现有实验."""
import os, glob, json, math, statistics, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

RES = "/home/fenghn/CvTest/experiments/results"
RUNS = "/home/fenghn/CvTest/experiments/runs"
ROOT = "/home/fenghn/CvTest/data/Dataset B"
FIG = "/home/fenghn/CvTest/experiments/figs_test"
os.makedirs(FIG, exist_ok=True)
NC, IOU_T = 21, 0.5

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.8, "savefig.bbox": "tight"})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#999999"}


def corr(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx*sy else 0.0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


# ---------- per-class recall(JSON) + freq + size ----------
pc = json.load(open(f"{RES}/yolov8m_B_TEST.json"))["per_class"]
rec = {int(k): v["recall"] for k, v in pc.items()}
freq = {int(k): v["n_inst"] for k, v in pc.items()}
classes = sorted(rec.keys())
areas = collections.defaultdict(list)
for dd in glob.glob(ROOT + "/labels/*"):
    for f in glob.glob(dd + "/*.txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 5:
                areas[int(float(p[0]))].append(float(p[3]) * float(p[4]))
size = {c: (statistics.median(areas[c]) if areas[c] else 0) for c in classes}

# ---------- Fig1: recall vs freq / size ----------
ys = [rec[c] for c in classes]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax[0].scatter([freq[c] for c in classes], ys, c=CB["blue"], s=24, edgecolor="k", linewidth=0.3, zorder=3)
ax[0].set_xscale("log"); ax[0].set_xlabel("Class frequency (# instances, log)"); ax[0].set_ylabel("Per-class recall")
ax[0].set_title(f"(a) Recall vs frequency  ($r={corr([freq[c] for c in classes], ys):+.2f}$)")
ax[1].scatter([size[c]*100 for c in classes], ys, c=CB["orange"], s=24, edgecolor="k", linewidth=0.3, zorder=3)
ax[1].set_xscale("log"); ax[1].set_xlabel("Median object area (% of image, log)")
ax[1].set_title(f"(b) Recall vs object size  ($r={corr([size[c] for c in classes], ys):+.2f}$)")
for a in ax:
    a.set_ylim(0, 1); a.grid(True, ls=":", lw=0.5, alpha=0.6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_recall_vs_freq_size.pdf"); fig.savefig(f"{FIG}/fig1_recall_vs_freq_size.png", dpi=200)
plt.close(fig)

# ---------- matching pass (conf=0.001) for Fig2/Fig3 ----------
m = YOLO(f"{RUNS}/yolov8m_B_ts/weights/best.pt")
same_conf = {c: [] for c in range(NC)}
dec = {c: [0, 0, 0] for c in range(NC)}      # correct / confused / missed @0.25
for r in m.predict(source=ROOT+"/images/test2", conf=0.001, iou=0.6, stream=True, verbose=False, device=7):
    H, W = r.orig_shape
    stem = os.path.splitext(os.path.basename(r.path))[0]
    lbl = os.path.join(ROOT, "labels/test2", stem + ".txt")
    gts = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5:
                c = int(float(p[0])); x, y, w, h = map(float, p[1:5])
                gts.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
    cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); bx = r.boxes.xyxy.cpu().numpy()
    for gc, gb in gts:
        sc, other25 = 0.0, False
        for k in range(len(cf)):
            if iou(bx[k], gb) >= IOU_T:
                if cl[k] == gc:
                    sc = max(sc, float(cf[k]))
                elif cf[k] >= 0.25:
                    other25 = True
        same_conf[gc].append(sc)
        dec[gc][0 if sc >= 0.25 else (1 if other25 else 2)] += 1

# ---------- Fig2: error decomposition ----------
order = sorted([c for c in range(NC) if sum(dec[c]) > 0], key=lambda c: dec[c][0]/sum(dec[c]))
cf_ = [dec[c][0]/sum(dec[c]) for c in order]
cn_ = [dec[c][1]/sum(dec[c]) for c in order]
mi_ = [dec[c][2]/sum(dec[c]) for c in order]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
lab = [str(c) for c in order]
ax.bar(lab, cf_, color=CB["green"], label="Correct", width=0.8)
ax.bar(lab, cn_, bottom=cf_, color=CB["orange"], label="Confused (other class)", width=0.8)
ax.bar(lab, mi_, bottom=[a+b for a, b in zip(cf_, cn_)], color=CB["red"], label="Missed (background)", width=0.8)
ax.set_xlabel("Class (sorted by recall, ascending)"); ax.set_ylabel("Fraction of GT instances"); ax.set_ylim(0, 1)
ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_error_decomposition.pdf"); fig.savefig(f"{FIG}/fig2_error_decomposition.png", dpi=200)
plt.close(fig)

# ---------- Fig3: recall headroom (R@0.25 -> R@max) ----------
R25 = {c: (float(np.mean(np.array(same_conf[c]) >= 0.25)) if same_conf[c] else 0) for c in range(NC)}
Rmx = {c: (float(np.mean(np.array(same_conf[c]) > 0.0)) if same_conf[c] else 0) for c in range(NC)}
order = sorted([c for c in range(NC) if same_conf[c]], key=lambda c: R25[c])
fig, ax = plt.subplots(figsize=(7.0, 3.0))
for i, c in enumerate(order):
    ax.plot([R25[c], Rmx[c]], [i, i], color=CB["grey"], lw=1.2, zorder=1)
ax.scatter([R25[c] for c in order], range(len(order)), c=CB["blue"], s=26, zorder=3, label="Recall @ conf=0.25 (default)")
ax.scatter([Rmx[c] for c in order], range(len(order)), c=CB["red"], s=26, zorder=3, label="Recall @ conf$\\to$0 (max)")
ax.set_yticks(range(len(order))); ax.set_yticklabels([str(c) for c in order], fontsize=7)
ax.set_xlabel("Per-class recall"); ax.set_ylabel("Class"); ax.set_xlim(0, 1)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9); ax.grid(True, axis="x", ls=":", lw=0.5, alpha=0.6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_recall_headroom.pdf"); fig.savefig(f"{FIG}/fig3_recall_headroom.png", dpi=200)
plt.close(fig)

print("figs saved to", FIG)
print("Fig1 corr: recall-freq =", round(corr([freq[c] for c in classes], ys), 3),
      "| recall-size =", round(corr([size[c] for c in classes], ys), 3))
hard = [c for c in range(NC) if same_conf[c] and R25[c] < 0.35]
print(f"难类 R@0.25={np.mean([R25[c] for c in hard]):.2f} -> R@max={np.mean([Rmx[c] for c in hard]):.2f}")
print("done")
