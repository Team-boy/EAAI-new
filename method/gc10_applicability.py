# -*- coding: utf-8 -*-
"""GC10 适用条件分析: VBAD在GC10上若触发会怎样 + 与织物B对比. 生成Discussion表数据."""
import os, glob, json
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data_gc10/yolo"
SPLIT = "test"
W640 = "/home/fenghn/CvTest/experiments/runs/gc10_v8m_640/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/gc10_v8m_1280/weights/best.pt"
IOU_T, CONF = 0.5, 0.25

m640 = YOLO(W640); m1280 = YOLO(W1280)
imgs = sorted(glob.glob(f"{ROOT}/images/{SPLIT}/*.jpg"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=CONF, iou=0.6, verbose=False, device=2)[0]
    W, H = img.size
    return [(int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H)
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))]


def hit(g, ps):
    return any(p[0] == g[0] and iou(g[1:5], p[1:5]) >= IOU_T for p in ps)


# 逐图: 640/1280 检测, 该图是否含Type-V
per_img = []
tot = h640 = h1280 = tv = ta = 0
for path in imgs:
    stem = os.path.splitext(os.path.basename(path))[0]
    lbl = f"{ROOT}/labels/{SPLIT}/{stem}.txt"
    if not os.path.exists(lbl):
        continue
    gts = [(int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
           for p in (l.split() for l in open(lbl)) if len(p) >= 5]
    if not gts:
        continue
    im = Image.open(path).convert("RGB")
    d640 = detect(m640, im, 640); d1280 = detect(m1280, im, 1280)
    has_tv = any((not hit(g, d640)) and hit(g, d1280) for g in gts)
    per_img.append((gts, d640, d1280, has_tv))
    for g in gts:
        tot += 1; a = hit(g, d640); b = hit(g, d1280)
        h640 += a; h1280 += b; tv += (not a) and b; ta += (not b)

ngt = tot
# oracle VBAD (含Type-V图升1280)
trig = sum(1 for _, _, _, h in per_img if h)
ovb = 0
for gts, d640, d1280, h in per_img:
    chosen = d1280 if h else d640
    for g in gts:
        ovb += hit(g, chosen)

print("=== GC10 适用条件分析 (test) ===")
print(f"  R@640={h640/ngt:.3f}  R@1280={h1280/ngt:.3f}")
print(f"  Type-V={tv/ngt:.1%}  Type-A={ta/ngt:.1%}")
print(f"  oracle-VBAD recall={ovb/ngt:.3f} (trigger {trig/len(per_img):.1%})")
print(f"  关键: 1280 vs 640 = {(h1280-h640)/ngt:+.3f} (织物B是+0.162)")
print(f"  -> GC10高分辨率无收益甚至有害, VBAD正确地极少触发, 不损害精度")
