# -*- coding: utf-8 -*-
"""训练 VBAD visibility head (640特征+标量 -> Type-V), test2评 learned触发recall+触发率.
对比: 全640=0.410, 全1280=0.572, oracle=0.615(29%), 简单规则~0.50."""
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

CACHE = "/home/fenghn/CvTest/method/vbad_cache"
IOU_T = 0.5
dev = "cuda:0"


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

F = np.stack([r["feat"] for r in tr]); S = np.stack([r["scalars"] for r in tr])
mu_f, sd_f = F.mean(0), F.std(0)+1e-6
mu_s, sd_s = S.mean(0), S.std(0)+1e-6


def build(rows):
    X = [np.concatenate([(r["feat"]-mu_f)/sd_f, (r["scalars"]-mu_s)/sd_s]) for r in rows]
    return np.array(X, np.float32), np.array([r["typeV"] for r in rows], np.float32)


Xtr, ytr = build(tr); Xte, yte = build(te)
net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 1)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
pos_w = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=dev)
lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
Xte_t = torch.tensor(Xte, device=dev)

for ep in range(300):
    net.train(); opt.zero_grad()
    loss = lossfn(net(Xtr_t).squeeze(1), ytr_t)
    loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    score = torch.sigmoid(net(Xte_t).squeeze(1)).cpu().numpy()

ngt = sum(len(r["gts"]) for r in te)


def recall_at(use_flags):
    h = 0
    for r, u in zip(te, use_flags):
        chosen = r["d1280"] if u else r["d640"]
        for g in r["gts"]:
            h += hit(g, chosen)
    return h/ngt


print("=== VBAD learned head (test2) ===")
print(f"  全640={recall_at([False]*len(te)):.3f}  全1280={recall_at([True]*len(te)):.3f}")
print(f"  ORACLE={recall_at([r['typeV']==1 for r in te]):.3f} (触发{np.mean([r['typeV'] for r in te]):.1%})")
print("  --- learned head 不同触发阈值 ---")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    flags = score >= thr
    print(f"  head>={thr}: recall={recall_at(flags):.3f} 触发{flags.mean():.1%}")
print(f"\n  head 预测 Type-V 的 AUC = {roc_auc_score(yte, score):.3f}")
