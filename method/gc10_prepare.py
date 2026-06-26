# -*- coding: utf-8 -*-
"""GC10-DET: VOC-XML -> YOLO 格式 + 分层切 train/val/test. 生成 data_split.yaml."""
import os, glob, random, shutil, re
import xml.etree.ElementTree as ET
import collections

SRC = "/home/fenghn/CvTest/data_gc10/gc10_extracted"
DST = "/home/fenghn/CvTest/data_gc10/yolo"
random.seed(0)

# 10 类名 (从 XML object name 的前缀 1-10 映射, name 形如 "3_yueyawan")
# 收集所有出现的类名 -> 排序映射到 0..9
xmls = sorted(glob.glob(f"{SRC}/lable/*.xml"))
print(f"XML 总数: {len(xmls)}")

# 第一遍: 收集类名
classnames = set()
for x in xmls:
    try:
        root = ET.parse(x).getroot()
        for o in root.findall("object"):
            nm = o.find("name").text.strip()
            m = re.match(r"^(\d+)", nm)
            if m:
                classnames.add(int(m.group(1)))
    except Exception:
        pass
# 按数字前缀(1..10) 映射到 0..9
ids = sorted(classnames)
id2idx = {gid: i for i, gid in enumerate(ids)}
classnames = [f"class{g}" for g in ids]
name2id = None  # 用 prefix 解析, 见下
print("类别:", classnames)

# 建图像名->路径索引(图像在 1..10 子目录)
img_index = {}
for d in [str(i) for i in range(1, 11)]:
    for p in glob.glob(f"{SRC}/{d}/*.jpg"):
        img_index[os.path.basename(p)] = p

# 第二遍: 转换每个 XML
samples = []   # (img_path, [(cls,cx,cy,w,h)], main_cls)
miss = 0
for x in xmls:
    try:
        root = ET.parse(x).getroot()
    except Exception:
        continue
    fn = root.find("filename").text.strip()
    if fn not in img_index:
        miss += 1
        continue
    W = float(root.find("size/width").text); H = float(root.find("size/height").text)
    boxes = []
    cls_count = collections.Counter()
    for o in root.findall("object"):
        nm = o.find("name").text.strip()
        mm = re.match(r"^(\d+)", nm)
        if not mm:
            continue
        gid = int(mm.group(1))
        if gid not in id2idx:
            continue
        c = id2idx[gid]
        b = o.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        cx = (x1+x2)/2/W; cy = (y1+y2)/2/H; bw = (x2-x1)/W; bh = (y2-y1)/H
        if bw > 0 and bh > 0:
            boxes.append((c, cx, cy, bw, bh)); cls_count[c] += 1
    if boxes:
        main_cls = cls_count.most_common(1)[0][0]
        samples.append((img_index[fn], boxes, main_cls))
print(f"有效样本: {len(samples)}, 图像缺失: {miss}")

# 分层切分: test 20%, val 15% of remaining, train 余下
by_cls = collections.defaultdict(list)
for s in samples:
    by_cls[s[2]].append(s)
train, val, test = [], [], []
for c, lst in by_cls.items():
    random.shuffle(lst)
    n = len(lst)
    nt = max(1, int(n*0.2)); nv = max(1, int(n*0.15))
    test += lst[:nt]; val += lst[nt:nt+nv]; train += lst[nt+nv:]
print(f"train {len(train)} / val {len(val)} / test {len(test)}")

# 写出 YOLO 格式
for split, data in [("train", train), ("val", val), ("test", test)]:
    idir, ldir = f"{DST}/images/{split}", f"{DST}/labels/{split}"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    for img_path, boxes, _ in data:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        dst_img = f"{idir}/{stem}.jpg"
        if not os.path.exists(dst_img):
            os.symlink(img_path, dst_img)
        with open(f"{ldir}/{stem}.txt", "w") as f:
            for c, cx, cy, bw, bh in boxes:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# yaml
import yaml
cfg = {"path": DST, "train": "images/train", "val": "images/val", "test": "images/test",
       "nc": len(classnames), "names": classnames}
yaml.safe_dump(cfg, open(f"{DST}/data_split.yaml", "w"), allow_unicode=True, sort_keys=False)
print(f"yaml: {DST}/data_split.yaml")

# 实例分布(看长尾)
inst = collections.Counter()
for _, boxes, _ in samples:
    for c, *_ in boxes:
        inst[c] += 1
total = sum(inst.values())
print("\n实例分布(长尾检查):")
for c in sorted(inst, key=lambda k: -inst[k]):
    print(f"  {classnames[c]}: {inst[c]} ({100*inst[c]/total:.1f}%)")
print(f"不均衡比: {max(inst.values())/min(inst.values()):.1f}x")
