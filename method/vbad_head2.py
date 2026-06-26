# -*- coding: utf-8 -*-
"""VBAD v2 空间 head: 小CNN处理 (3,20,20) 空间特征 -> Type-V.
目标: AUC从0.65提升, recall逼近oracle 0.615."""
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

CACHE = "/home/fenghn/CvTest/method/vbad_cache2"
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

Xtr = np.stack([r["spatial"] for r in tr]); ytr = np.array([r["typeV"] for r in tr], np.float32)
Xte = np.stack([r["spatial"] for r in te]); yte = np.array([r["typeV"] for r in te], np.float32)
# 逐通道标准化
mu = Xtr.mean((0, 2, 3), keepdims=True); sd = Xtr.std((0, 2, 3), keepdims=True)+1e-6
Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd


class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveMaxPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*16, 64), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(self.conv(x))


net = SpatialHead().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
pos_w = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=dev)
lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
ytr_t = torch.tensor(ytr, device=dev)
Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)

bs = 128
n = len(Xtr_t)
for ep in range(120):
    net.train()
    perm = torch.randperm(n, device=dev)
    for i in range(0, n, bs):
        idx = perm[i:i+bs]
        opt.zero_grad()
        loss = lossfn(net(Xtr_t[idx]).squeeze(1), ytr_t[idx])
        loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    score = torch.sigmoid(net(Xte_t).squeeze(1)).cpu().numpy()

ngt = sum(len(r["gts"]) for r in te)


def recall_at(flags):
    h = 0
    for r, u in zip(te, flags):
        chosen = r["d1280"] if u else r["d640"]
        for g in r["gts"]:
            h += hit(g, chosen)
    return h/ngt


print("=== VBAD v2 spatial head (test2) ===")
print(f"  全640={recall_at([False]*len(te)):.3f}  全1280={recall_at([True]*len(te)):.3f}")
print(f"  ORACLE={recall_at([r['typeV']==1 for r in te]):.3f} (触发{np.mean(yte):.1%})")
print(f"  v1 GAP head: AUC=0.652, best recall 0.53@45%")
print("  --- v2 spatial head 不同触发阈值 ---")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    flags = score >= thr
    print(f"  head>={thr}: recall={recall_at(flags):.3f} 触发{flags.mean():.1%}")
print(f"\n  v2 head AUC = {roc_auc_score(yte, score):.3f}")
