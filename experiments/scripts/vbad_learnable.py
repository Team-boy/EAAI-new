# -*- coding: utf-8 -*-
"""VBAD 可学性验证: 不用GT, 用640可获得的信号做触发, 对比oracle(29%触发/0.615)."""
import os, glob
from ultralytics import YOLO
from PIL import Image
import numpy as np

ROOT = "/home/fenghn/CvTest/data/Dataset B"
SPLIT = "test2"
W640 = "/home/fenghn/CvTest/experiments/runs/yolov8m_B_ts/weights/best.pt"
W1280 = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
IOU_T = 0.5

m640 = YOLO(W640); m1280 = YOLO(W1280)
imgs = sorted(p for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
              for p in glob.glob(f"{ROOT}/images/{SPLIT}/{e}"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua > 0 else 0


def detect(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=0.6, verbose=False, device=0)[0]
    W, H = img.size
    out = []
    for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                        r.boxes.conf.cpu().numpy()):
        out.append((int(c), (b[0]+b[2])/2/W, (b[1]+b[3])/2/H, (b[2]-b[0])/W, (b[3]-b[1])/H, float(cf)))
    return out


def hit(gt, preds):
    gc, gx, gy, gw, gh = gt
    return any(pc == gc and iou((gx, gy, gw, gh), (px, py, pw, ph)) >= IOU_T
               for pc, px, py, pw, ph, _ in preds)


data = []
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
    d640 = detect(m640, im, 640, 0.25)
    d640_lo = detect(m640, im, 640, 0.05)
    d1280 = detect(m1280, im, 1280, 0.25)
    has_typeV = any((not hit(g, d640)) and hit(g, d1280) for g in gts)
    lo_extra = len([p for p in d640_lo if p[5] < 0.25])
    min_box = min([w*h for _, _, _, w, h, _ in d640], default=1.0)
    mean_conf = np.mean([p[5] for p in d640]) if d640 else 0.0
    data.append({"gts": gts, "d640": d640, "d1280": d1280, "typeV": has_typeV,
                 "lo_extra": lo_extra, "min_box": min_box, "mean_conf": mean_conf})

ngt = sum(len(d["gts"]) for d in data)


def ev(trigger_fn):
    hit_c = trig = 0
    for d in data:
        use = trigger_fn(d)
        trig += use
        chosen = d["d1280"] if use else d["d640"]
        for g in d["gts"]:
            hit_c += hit(g, chosen)
    return hit_c/ngt, trig/len(data)


r640, _ = ev(lambda d: False)
r1280, _ = ev(lambda d: True)
roracle, torcl = ev(lambda d: d["typeV"])
print("=== 参照 ===")
print(f"  全640={r640:.3f}  全1280={r1280:.3f}(触发100%)  ORACLE={roracle:.3f}(触发{torcl:.1%})")
print("\n=== 真实信号触发(只用640可得) ===")
for thr in [1, 2, 3]:
    r, t = ev(lambda d, T=thr: d["lo_extra"] >= T)
    print(f"  低conf额外检测>={thr}: recall={r:.3f} 触发{t:.1%}")
mb = sorted(d["min_box"] for d in data)
for q in [0.3, 0.5]:
    v = mb[int(len(mb)*q)]
    r, t = ev(lambda d, V=v: d["min_box"] <= V)
    print(f"  最小框<=q{q}: recall={r:.3f} 触发{t:.1%}")
mc = sorted(d["mean_conf"] for d in data)
for q in [0.3, 0.5]:
    v = mc[int(len(mc)*q)]
    r, t = ev(lambda d, V=v: d["mean_conf"] <= V)
    print(f"  平均conf<=q{q}: recall={r:.3f} 触发{t:.1%}")
r, t = ev(lambda d: d["lo_extra"] >= 2 or d["min_box"] < mb[int(len(mb)*0.2)])
print(f"  组合(低conf>=2 或 小框): recall={r:.3f} 触发{t:.1%}")

tv = [d["lo_extra"] for d in data if d["typeV"]]
ntv = [d["lo_extra"] for d in data if not d["typeV"]]
print("\n=== 可学性核心: 信号能否区分Type-V图 ===")
print(f"  Type-V图 低conf额外检测均值={np.mean(tv):.2f}  非Type-V={np.mean(ntv):.2f}")
print(f"  Type-V图 平均conf={np.mean([d['mean_conf'] for d in data if d['typeV']]):.3f}  非={np.mean([d['mean_conf'] for d in data if not d['typeV']]):.3f}")
