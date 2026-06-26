#!/usr/bin/env python
"""Route1 test-split: 原 val 冻结为 test; train 按类分层切 15% 为 val'.
生成新 yaml (train2/val2/test2) 指向软链目录, 不动原始数据. 确定性 seed=0."""
import os, glob, random, collections, shutil, yaml, argparse

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)         # .../Dataset B
p.add_argument("--train", default="train")      # 原 train 子目录名
p.add_argument("--val", default="val")          # 原 val 子目录名 -> 冻结为 test
p.add_argument("--frac", type=float, default=0.15)
args = p.parse_args()
random.seed(0)
ROOT = args.root
IMG = f"{ROOT}/images"; LBL = f"{ROOT}/labels"

def imgs(split):
    out = []
    for e in ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp"):
        out += glob.glob(f"{IMG}/{split}/{e}")
    return sorted(out)

train_imgs = imgs(args.train)
# 每张图的"主类"(出现最多的类)用于分层
def main_cls(ip):
    stem = os.path.splitext(os.path.basename(ip))[0]
    lp = f"{LBL}/{args.train}/{stem}.txt"
    c = collections.Counter()
    if os.path.exists(lp):
        for line in open(lp):
            s = line.split()
            if s: c[int(float(s[0]))] += 1
    return c.most_common(1)[0][0] if c else -1

by = collections.defaultdict(list)
for ip in train_imgs:
    by[main_cls(ip)].append(ip)

val2, train2 = [], []
for c, lst in by.items():
    random.shuffle(lst)
    k = max(1, int(len(lst) * args.frac)) if len(lst) > 3 else 0
    val2 += lst[:k]; train2 += lst[k:]

def link_split(name, img_list, src_split):
    di, dl = f"{IMG}/{name}", f"{LBL}/{name}"
    for d in (di, dl):
        if os.path.islink(d) or os.path.exists(d):
            if os.path.islink(d): os.unlink(d)
            else: shutil.rmtree(d)
        os.makedirs(d)
    for ip in img_list:
        stem = os.path.splitext(os.path.basename(ip))[0]
        os.symlink(ip, f"{di}/{os.path.basename(ip)}")
        lp = f"{LBL}/{src_split}/{stem}.txt"
        if os.path.exists(lp): os.symlink(lp, f"{dl}/{stem}.txt")

link_split("train2", train2, args.train)
link_split("val2", val2, args.train)
# test2 = 原 val 冻结
link_split("test2", imgs(args.val), args.val)

# 新 yaml
cfg = yaml.safe_load(open(glob.glob(f"{ROOT}/*.yaml")[0]))
cfg["path"] = ROOT
cfg["train"] = "images/train2"; cfg["val"] = "images/val2"; cfg["test"] = "images/test2"
out_yaml = f"{ROOT}/data_split.yaml"
yaml.safe_dump(cfg, open(out_yaml, "w"), allow_unicode=True, sort_keys=False)
print(f"{os.path.basename(ROOT)}: train2={len(train2)} val2={len(val2)} test2(=orig val)={len(imgs(args.val))}")
print("  yaml:", out_yaml)
