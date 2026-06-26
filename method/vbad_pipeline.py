# -*- coding: utf-8 -*-
"""VBAD 正式推理 pipeline (可复现).
640快筛 -> visibility head 打分 -> 高分图升全图1280复检 -> 合并.
detector frozen; visibility head 由 cross-resolution failure (Type-V) 监督.
用法: python vbad_pipeline.py --head method/runs/vbad_head.pt --tau 0.4 --device 2"""
import argparse, os, glob, pickle
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
IOU_T, CONF = 0.5, 0.25
FEAT_LAYER = 9


class VisibilityHead(nn.Module):
    """640 全局特征(576) + 4标量 -> Type-V 概率."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def hit(gt, preds):
    return any(p[0] == gt[0] and iou(gt[1:5], p[1:5]) >= IOU_T for p in preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.4, help="visibility 触发阈值")
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="test2")
    ap.add_argument("--train_cache", default="/home/fenghn/CvTest/method/vbad_cache/train2.pkl")
    ap.add_argument("--test_cache", default="/home/fenghn/CvTest/method/vbad_cache/test2.pkl")
    a = ap.parse_args()
    dev = f"cuda:{a.device}"

    # 1) 训 visibility head (从已导出的特征缓存, detector frozen)
    tr = pickle.load(open(a.train_cache, "rb"))
    te = pickle.load(open(a.test_cache, "rb"))
    F = np.stack([r["feat"] for r in tr]); S = np.stack([r["scalars"] for r in tr])
    mu_f, sd_f = F.mean(0), F.std(0)+1e-6; mu_s, sd_s = S.mean(0), S.std(0)+1e-6
    def feat(rows):
        return np.array([np.concatenate([(r["feat"]-mu_f)/sd_f, (r["scalars"]-mu_s)/sd_s]) for r in rows], np.float32)
    Xtr, ytr = feat(tr), np.array([r["typeV"] for r in tr], np.float32)
    Xte = feat(te)
    head = VisibilityHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=dev)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    for _ in range(300):
        head.train(); opt.zero_grad(); lf(head(Xt).squeeze(1), yt).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        s_vis = torch.sigmoid(head(torch.tensor(Xte, device=dev)).squeeze(1)).cpu().numpy()
    os.makedirs("/home/fenghn/CvTest/method/runs", exist_ok=True)
    torch.save(head.state_dict(), "/home/fenghn/CvTest/method/runs/vbad_head.pt")

    # 2) VBAD 分诊推理(用缓存的 d640/d1280; 真实部署时 d1280 仅在触发时计算)
    ngt = sum(len(r["gts"]) for r in te)
    hit_c = trig = 0
    for r, sv in zip(te, s_vis):
        use = sv >= a.tau
        trig += use
        chosen = r["d1280"] if use else r["d640"]
        for g in r["gts"]:
            hit_c += hit(g, chosen)
    recall = hit_c/ngt
    trig_rate = trig/len(te)
    # 算力: 640对所有图 + 1280对触发图 (1280≈4x FLOPs of 640)
    rel_flops = 1.0 + trig_rate*4.0
    print(f"=== VBAD pipeline (tau={a.tau}) ===")
    print(f"  recall = {recall:.3f}")
    print(f"  trigger rate = {trig_rate:.1%}")
    print(f"  relative FLOPs = {rel_flops:.2f}x (vs full-1280 = 4.0x)")
    print(f"  ref: 640={0.410} full1280={0.572} oracle=0.615")
    print(f"  head saved -> method/runs/vbad_head.pt")


if __name__ == "__main__":
    main()
