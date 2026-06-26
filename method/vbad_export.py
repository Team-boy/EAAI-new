# -*- coding: utf-8 -*-
"""导出 VBAD head 训练数据: 每图的 640 backbone 特征 + 多信号 + Type-V 标签 + 检测缓存.
train2 训 head, test2 评. 用现有 640/1280 模型."""
import os, glob, pickle
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
OUT = "/home/fenghn/CvTest/method/vbad_cache"
IOU_T = 0.5
FEAT_LAYER = 9   # SPPF 输出, 全局特征

os.makedirs(OUT, exist_ok=True)
m640 = YOLO(W640); m1280 = YOLO(W1280)

# hook 取 640 backbone 全局特征
feat_store = {}
def hook(m, i, o): feat_store["f"] = o.detach()
m640.model.model[FEAT_LAYER].register_forward_hook(hook)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, device=0)[0]
    W, H = img.size
    return [(int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H, float(cf))
            for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy())]


def hit(gt, preds):
    return any(p[0] == gt[0] and iou(gt[1:5], p[1:5]) >= IOU_T for p in preds)


def process(split):
    imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
                  for p in glob.glob(f"{ROOT}/images/{split}/{e}"))
    rows = []
    for path in imgs:
        stem = os.path.splitext(os.path.basename(path))[0]
        lbl = f"{ROOT}/labels/{split}/{stem}.txt"
        if not os.path.exists(lbl):
            continue
        gts = [(int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
               for p in (l.split() for l in open(lbl)) if len(p) >= 5]
        if not gts:
            continue
        im = Image.open(path).convert("RGB")
        feat_store.clear()
        d640 = detect(m640, im, 640, 0.25)          # 触发 hook
        feat = feat_store["f"].float().mean(dim=(2, 3)).squeeze(0).cpu().numpy()  # GAP -> (C,)
        d640_lo = detect(m640, im, 640, 0.05)
        d1280 = detect(m1280, im, 1280, 0.25)
        typeV = int(any((not hit(g, d640)) and hit(g, d1280) for g in gts))
        scalars = np.array([
            len([p for p in d640_lo if p[5] < 0.25]),                 # 低conf额外检测数
            len(d640),                                                # 检测数
            min([w*h for _, _, _, w, h, _ in d640], default=1.0),     # 最小框
            np.mean([p[5] for p in d640]) if d640 else 0.0,           # 平均conf
        ], dtype=np.float32)
        rows.append({"stem": stem, "feat": feat.astype(np.float32), "scalars": scalars,
                     "typeV": typeV, "gts": gts, "d640": d640, "d1280": d1280})
    pickle.dump(rows, open(f"{OUT}/{split}.pkl", "wb"))
    pos = sum(r["typeV"] for r in rows)
    print(f"{split}: {len(rows)}图, Type-V正例 {pos} ({pos/len(rows):.1%}), feat维度 {rows[0]['feat'].shape}")


process("train2")
process("test2")
print("EXPORT DONE")
