"""test2 上算密度分支 cls18 recall, 对比 baseline 0.08."""
import argparse, os, glob
import numpy as np
import torch
from method.density_net import DensityNet
from method.density_decode import heatmap_to_boxes
from method.region_eval import recall_at_iou
from method.density_dataset import CLS18
from PIL import Image

ROOT = "/home/fenghn/CvTest/data/Dataset B"


def gt_boxes(stem, split):
    lbl = f"{ROOT}/labels/{split}/{stem}.txt"
    out = []
    if os.path.exists(lbl):
        for line in open(lbl):
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) == CLS18:
                out.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/fenghn/CvTest/method/runs/density_mvp.pt")
    ap.add_argument("--split", default="test2")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--hm", type=int, default=160)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    dev = f"cuda:{a.device}"
    net = DensityNet(a.hm).to(dev)
    net.load_state_dict(torch.load(a.weights, map_location=dev)); net.eval()
    exts = ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
    imgs = sorted(p for e in exts for p in glob.glob(f"{ROOT}/images/{a.split}/{e}"))
    hit = total = 0
    with torch.no_grad():
        for path in imgs:
            stem = os.path.splitext(os.path.basename(path))[0]
            gts = gt_boxes(stem, a.split)
            if not gts:
                continue
            im = Image.open(path).convert("RGB").resize((a.img, a.img))
            x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
            hm = net(x)[0, 0].cpu().numpy()
            r = recall_at_iou(heatmap_to_boxes(hm, thr=a.thr), gts, 0.5)
            if r is not None:
                hit += r * len(gts); total += len(gts)
    recall = hit / total if total else 0.0
    print(f"cls18 density recall@0.5 ({a.split}, thr={a.thr}) = {recall:.3f}  (baseline bbox=0.08)")
    print("MVP", "PASS" if recall >= 0.25 else "FAIL", f"(目标>=0.25 实测 {recall:.3f})")


if __name__ == "__main__":
    main()
