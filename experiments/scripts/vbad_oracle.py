# -*- coding: utf-8 -*-
"""VBAD oracle: image-level 分辨率分诊上界 + 触发率 + Type-A(不可救)比例.
Type-V图(含 640漏&1280救 的GT)用1280, 其余640. 用现有模型纯推理."""
import os, glob, collections
from ultralytics import YOLO
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"
SPLIT = "test2"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
IOU_T, CONF = 0.5, 0.25
SMALL = {7: "mao", 17: "anlun", 13: "weisuo", 10: "duanjing", 11: "diaojing", 16: "xingtiao"}

m640 = YOLO(W640); m1280 = YOLO(W1280)
imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
              for p in glob.glob(f"{ROOT}/images/{SPLIT}/{e}"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=CONF, iou=0.6, verbose=False, device=0)[0]
    W, H = img.size
    return [(int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H)
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))]


def gt_hit(gt, preds):
    gc, gx, gy, gw, gh = gt
    return any(pc == gc and iou((gx, gy, gw, gh), (px, py, pw, ph)) >= IOU_T
               for pc, px, py, pw, ph in preds)


n_img = trig = 0
gt_tot = r640 = r1280 = rvbad = 0
typeV = typeA = 0
sm_tot = sm_640 = sm_vbad = sm_A = 0

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
    d640 = detect(m640, im, 640)
    d1280 = detect(m1280, im, 1280)
    # 该图是否含 Type-V GT (640漏 & 1280救)
    has_typeV = any((not gt_hit(g, d640)) and gt_hit(g, d1280) for g in gts)
    n_img += 1
    if has_typeV:
        trig += 1
    chosen = d1280 if has_typeV else d640   # oracle 分诊
    for g in gts:
        gt_tot += 1
        h640 = gt_hit(g, d640); h1280 = gt_hit(g, d1280); hv = gt_hit(g, chosen)
        r640 += h640; r1280 += h1280; rvbad += hv
        if (not h640) and h1280:
            typeV += 1
        if not h1280:
            typeA += 1            # 连1280都救不了
        if g[0] in SMALL:
            sm_tot += 1; sm_640 += h640; sm_vbad += hv
            if not h1280:
                sm_A += 1

print(f"=== VBAD Oracle (image-level, {SPLIT}) ===")
print(f"  图数 {n_img}, GT数 {gt_tot}")
print(f"  触发率(升1280的图)  = {trig}/{n_img} = {trig/n_img:.1%}")
print(f"  640 recall          = {r640/gt_tot:.3f}")
print(f"  1280 full recall    = {r1280/gt_tot:.3f}")
print(f"  VBAD oracle recall  = {rvbad/gt_tot:.3f}")
print(f"  => VBAD vs 640: {(rvbad-r640)/gt_tot:+.3f}; 拿到1280增益的 {(rvbad-r640)/(r1280-r640)*100:.0f}%")
print(f"\n  Type-V (640漏1280救, VBAD能救): {typeV} ({typeV/gt_tot:.1%})")
print(f"  Type-A (1280也漏, 不可救天花板): {typeA} ({typeA/gt_tot:.1%})  <- recall根本上限")
print(f"\n  极小目标类: 640={sm_640/sm_tot:.3f} VBAD={sm_vbad/sm_tot:.3f}; 其中{sm_A}/{sm_tot}({sm_A/sm_tot:.0%})连1280也救不了")
