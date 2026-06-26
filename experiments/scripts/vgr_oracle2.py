# -*- coding: utf-8 -*-
"""VGR oracle 修正版: 640快筛 + 1280 teacher 复检漏检patch. 双模型方案."""
import os, glob, collections
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
SPLIT = "test2"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
IOU_T, CONF = 0.5, 0.25
SMALL = {7: "mao", 17: "anlun", 13: "weisuo", 10: "duanjing", 11: "diaojing", 16: "xingtiao"}

m640 = YOLO(W640)
m1280 = YOLO(W1280)
imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
              for p in glob.glob(f"{ROOT}/images/{SPLIT}/{e}"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img_pil, imgsz):
    r = model.predict(img_pil, imgsz=imgsz, conf=CONF, iou=0.6, verbose=False, device=0)[0]
    W, H = img_pil.size
    out = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        out.append((int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H))
    return out


def hit(gt, preds):
    gc, gx, gy, gw, gh = gt
    for pc, px, py, pw, ph in preds:
        if pc == gc and iou((gx, gy, gw, gh), (px, py, pw, ph)) >= IOU_T:
            return True
    return False


tot = full_hit = vgr_hit = 0
per_cls = collections.defaultdict(lambda: [0, 0, 0])

for path in imgs:
    stem = os.path.splitext(os.path.basename(path))[0]
    lbl = f"{ROOT}/labels/{SPLIT}/{stem}.txt"
    if not os.path.exists(lbl):
        continue
    gts = []
    for line in open(lbl):
        p = line.split()
        if len(p) >= 5:
            gts.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    if not gts:
        continue
    im = Image.open(path).convert("RGB")
    W, H = im.size
    full = detect(m640, im, 640)
    for gt in gts:
        gc, gx, gy, gw, gh = gt
        tot += 1
        per_cls[gc][0] += 1
        fh = hit(gt, full)
        if fh:
            full_hit += 1; vgr_hit += 1; per_cls[gc][1] += 1; per_cls[gc][2] += 1
        else:
            side = max(0.15, min(0.5, max(gw, gh) * 4))
            cxp, cyp = gx * W, gy * H
            half = side * max(W, H) / 2
            px1, py1 = max(0, cxp - half), max(0, cyp - half)
            px2, py2 = min(W, cxp + half), min(H, cyp + half)
            patch = im.crop((px1, py1, px2, py2))
            pw, ph = patch.size
            if pw < 10 or ph < 10:
                continue
            # teacher 复检, patch 放大到 1280
            mapped = []
            for pc, x, y, w, h in detect(m1280, patch, 1280):
                mapped.append((pc, (px1 + x * pw) / W, (py1 + y * ph) / H, w * pw / W, h * ph / H))
            if hit(gt, mapped):
                vgr_hit += 1; per_cls[gc][2] += 1

print(f"=== VGR Oracle v2 (teacher reinspect, {SPLIT}) ===")
print(f"  total GT: {tot}")
print(f"  640 full recall      = {full_hit/tot:.3f}")
print(f"  VGR oracle recall    = {vgr_hit/tot:.3f}  (640 union teacher-patch)")
print(f"  => teacher saved      = {(vgr_hit-full_hit)/tot:+.3f} ({vgr_hit-full_hit} GTs)")
print(f"  (ref: teacher full-img 1280 = 0.572)")
st = sum(per_cls[c][0] for c in SMALL if c in per_cls)
sf = sum(per_cls[c][1] for c in SMALL if c in per_cls)
sv = sum(per_cls[c][2] for c in SMALL if c in per_cls)
if st:
    print(f"\nsmall-defect: 640 recall={sf/st:.3f} -> VGR oracle={sv/st:.3f} (saved {sv-sf})")
    for c in sorted(SMALL):
        if c in per_cls and per_cls[c][0] > 0:
            t, f, v = per_cls[c]
            print(f"    cls{c}({SMALL[c]}): {f/t:.2f} -> {v/t:.2f}  (n={t})")
