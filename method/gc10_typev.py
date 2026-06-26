# -*- coding: utf-8 -*-
"""GC10: 算 Type-V (640漏∩1280救) / Type-A 比例, 判定是否适合做VBAD."""
import os, glob
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
    for g in gts:
        tot += 1
        a = hit(g, d640); b = hit(g, d1280)
        h640 += a; h1280 += b
        tv += (not a) and b
        ta += (not b)

print(f"=== GC10 Type-V/A 判定 (test, {tot} GT) ===")
print(f"  R@640 = {h640/tot:.3f}")
print(f"  R@1280 = {h1280/tot:.3f}")
print(f"  Type-V (640漏1280救) = {tv} ({tv/tot:.1%})")
print(f"  Type-A (1280也漏)    = {ta} ({ta/tot:.1%})")
print(f"\n  对照织物B: Type-V=21.5%, 高分辨率+8.5")
print(f"  判定: {'适合VBAD(Type-V显著)' if tv/tot > 0.10 else '不适合(GC10缺陷已充分可见, Type-V很少)'}")
