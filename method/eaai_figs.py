# -*- coding: utf-8 -*-
"""EAAI论文图: fig2 taxonomy堆叠柱, fig3 recall-cost曲线, fig5 visibility ROC."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score

CACHE = "/home/fenghn/CvTest/method/vbad_cache"
FIG = "/home/fenghn/CvTest/paper_eaai/figs"
os.makedirs(FIG, exist_ok=True)
IOU_T = 0.5
plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.bbox": "tight"})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "grey": "#999999"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def hit(gt, preds):
    return any(p[0] == gt[0] and iou(gt[1:5], p[1:5]) >= IOU_T for p in preds)


tr = pickle.load(open(f"{CACHE}/train2.pkl", "rb"))
te = pickle.load(open(f"{CACHE}/test2.pkl", "rb"))
ngt = sum(len(r["gts"]) for r in te)

F = np.stack([r["feat"] for r in tr]); S = np.stack([r["scalars"] for r in tr])
mu_f, sd_f = F.mean(0), F.std(0)+1e-6; mu_s, sd_s = S.mean(0), S.std(0)+1e-6
def bf(rows): return np.array([np.concatenate([(r["feat"]-mu_f)/sd_f, (r["scalars"]-mu_s)/sd_s]) for r in rows], np.float32)
Xtr = bf(tr); ytr = np.array([r["typeV"] for r in tr], np.float32); Xte = bf(te)
dev = "cuda:0"
net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=dev)
lf = nn.BCEWithLogitsLoss(pos_weight=pw)
Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
for _ in range(300):
    net.train(); opt.zero_grad(); lf(net(Xt).squeeze(1), yt).backward(); opt.step()
net.eval()
with torch.no_grad():
    s_vis = torch.sigmoid(net(torch.tensor(Xte, device=dev)).squeeze(1)).cpu().numpy()


def recall_cost(flags):
    h = sum(hit(g, (r["d1280"] if u else r["d640"])) for r, u in zip(te, flags) for g in r["gts"])
    return h/ngt, 1.0+np.mean(flags)*4.0


# Fig3 recall-cost
taus = np.linspace(0.1, 0.9, 17)
vb = [recall_cost(s_vis >= t) for t in taus]
r640, c640 = recall_cost([False]*len(te))
r1280, c1280 = recall_cost([True]*len(te))
roc_r, roc_c = recall_cost([r["typeV"] == 1 for r in te])
mb = np.array([r["scalars"][2] for r in te])
simp = [recall_cost(mb <= np.quantile(mb, q)) for q in np.linspace(0.1, 0.9, 9)]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot([x[1] for x in vb], [x[0] for x in vb], "-o", color=CB["blue"], ms=4, label="VBAD (learned head)", zorder=3)
ax.plot([x[1] for x in simp], [x[0] for x in simp], "--s", color=CB["grey"], ms=3, label="Simple trigger (min-box)")
ax.scatter([c640], [r640], color="k", marker="v", s=60, zorder=4, label="Low-res 640")
ax.scatter([c1280], [r1280], color=CB["red"], marker="^", s=60, zorder=4, label="Full high-res 1280")
ax.scatter([roc_c], [roc_r], color=CB["green"], marker="*", s=140, zorder=4, label="Oracle triage")
ax.set_xlabel("Relative computational cost ($\\times$ low-res FLOPs)")
ax.set_ylabel("Recall @ IoU=0.5")
ax.grid(True, ls=":", alpha=0.6); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_recall_cost.pdf"); fig.savefig(f"{FIG}/fig3_recall_cost.png", dpi=160); plt.close(fig)

# Fig2 taxonomy
SMALL = {7: "nep", 17: "spandex-br", 10: "broken-warp", 13: "weft-shrink", 11: "hanging-warp"}
groups = [("All", None)] + [(SMALL[c], c) for c in SMALL]
labels, N, V, A = [], [], [], []
for name, c in groups:
    tn = tv = ta = tot = 0
    for r in te:
        for g in r["gts"]:
            if c is not None and g[0] != c:
                continue
            tot += 1
            a = hit(g, r["d640"]); b = hit(g, r["d1280"])
            tn += a; tv += (not a) and b; ta += (not b)
    if tot:
        labels.append(name); N.append(tn/tot); V.append(tv/tot); A.append(ta/tot)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
x = range(len(labels))
ax.bar(x, N, color=CB["green"], label="Type-N (detected @640)")
ax.bar(x, V, bottom=N, color=CB["orange"], label="Type-V (recovered @1280)")
ax.bar(x, A, bottom=[n+v for n, v in zip(N, V)], color=CB["red"], label="Type-A (unrecoverable)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Fraction of GT defects"); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper center", ncol=1)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_taxonomy.pdf"); fig.savefig(f"{FIG}/fig2_taxonomy.png", dpi=160); plt.close(fig)

# Fig5 ROC
yte = np.array([r["typeV"] for r in te])
fpr, tpr, _ = roc_curve(yte, s_vis); auc = roc_auc_score(yte, s_vis)
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.plot(fpr, tpr, color=CB["blue"], lw=2, label=f"Visibility head (AUC={auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color=CB["grey"], lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, ls=":", alpha=0.6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_roc.pdf"); fig.savefig(f"{FIG}/fig5_roc.png", dpi=160); plt.close(fig)

print("figs:", sorted(os.listdir(FIG)))
print(f"fig3: 640=({c640:.1f},{r640:.3f}) 1280=({c1280:.1f},{r1280:.3f}) oracle=({roc_c:.1f},{roc_r:.3f})")
print(f"fig2 All: N={N[0]:.2f} V={V[0]:.2f} A={A[0]:.2f} | fig5 AUC={auc:.3f}")
