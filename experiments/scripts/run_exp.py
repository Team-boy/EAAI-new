#!/usr/bin/env python
"""统一实验入口: 训练 + 长尾分组评测 (head/med/tail by instance freq, 自动剔除空类)."""
import argparse, json, os, glob
from collections import Counter
import yaml


def class_freq(data_yaml):
    root = os.path.dirname(os.path.abspath(data_yaml))
    cfg = yaml.safe_load(open(data_yaml))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    nc = len(names)
    cnt = Counter()
    for d in glob.glob(os.path.join(root, "labels", "*")):
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.txt")):
            for line in open(f):
                s = line.strip()
                if s:
                    cnt[int(float(s.split()[0]))] += 1
    return names, nc, cnt


def groups(cnt, nc, head=0.10, tail=0.02):
    total = sum(cnt.values()) or 1
    H, M, T = [], [], []
    for c in range(nc):
        n = cnt.get(c, 0)
        if n == 0:
            continue                       # 空类剔除(不参与指标)
        frac = n / total
        (H if frac >= head else T if frac <= tail else M).append(c)
    return H, M, T


def evaluate(model_obj, data, device, names, cnt, nc):
    res = model_obj.val(data=data, split="val", device=device, verbose=False, plots=False)
    box = res.box
    idx = [int(c) for c in box.ap_class_index]
    rec = {idx[i]: float(box.r[i]) for i in range(len(idx))}
    ap50 = {idx[i]: float(box.ap50[i]) for i in range(len(idx))}
    ap = {idx[i]: float(box.ap[i]) for i in range(len(idx))}
    H, M, T = groups(cnt, nc)

    def agg(group, d):
        v = [d.get(c, 0.0) for c in group]
        return sum(v) / len(v) if v else 0.0

    return {
        "overall": {"mAP50": float(box.map50), "mAP50_95": float(box.map),
                    "recall_mean": float(box.mr), "precision_mean": float(box.mp)},
        "groups": {g: {"classes": grp, "n_cls": len(grp),
                       "recall": agg(grp, rec), "mAP50": agg(grp, ap50), "mAP50_95": agg(grp, ap)}
                   for g, grp in (("head", H), ("med", M), ("tail", T))},
        "per_class": {c: {"name": str(names[c]), "n_inst": cnt.get(c, 0),
                          "recall": rec.get(c, 0.0), "ap50": ap50.get(c, 0.0)} for c in idx},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="/home/fenghn/CvTest/experiments/runs")
    p.add_argument("--results", default="/home/fenghn/CvTest/experiments/results")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights", default=None)
    a = p.parse_args()

    from ultralytics import YOLO, RTDETR
    ref = (a.weights or a.model).lower()
    Model = RTDETR if "rtdetr" in ref else YOLO
    names, nc, cnt = class_freq(a.data)

    if a.eval_only:
        out = evaluate(Model(a.weights), a.data, a.device, names, cnt, nc)
    else:
        m = Model(a.model)
        m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, seed=a.seed, project=a.project, name=a.name,
                exist_ok=True, verbose=False, plots=True, workers=8)
        best = os.path.join(a.project, a.name, "weights", "best.pt")
        out = evaluate(Model(best), a.data, a.device, names, cnt, nc)

    out["meta"] = {"model": a.model, "data": os.path.basename(os.path.dirname(a.data)),
                   "name": a.name, "epochs": a.epochs}
    os.makedirs(a.results, exist_ok=True)
    jpath = os.path.join(a.results, a.name + ".json")
    json.dump(out, open(jpath, "w"), indent=2, ensure_ascii=False)

    o, g = out["overall"], out["groups"]
    print(f"\n=== {a.name} ===  overall mAP50={o['mAP50']:.4f}  mAP50-95={o['mAP50_95']:.4f}  meanR={o['recall_mean']:.4f}")
    for k in ("head", "med", "tail"):
        gg = g[k]
        print(f"  {k:4} ({gg['n_cls']:>2} cls): Recall={gg['recall']:.4f}  mAP50={gg['mAP50']:.4f}  mAP50-95={gg['mAP50_95']:.4f}")
    print("saved:", jpath)


if __name__ == "__main__":
    main()
