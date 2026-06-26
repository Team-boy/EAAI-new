# -*- coding: utf-8 -*-
"""⑤ 标注质量诊断: 高漏检类的框大小/每图实例/异常值."""
import glob, os, collections, statistics
ROOT = "/home/fenghn/CvTest/data/Dataset B"
HARD = {7: "毛粒", 18: "稀密档", 13: "纬缩", 10: "断经", 17: "断氨纶", 11: "吊经", 16: "星跳跳花"}

# 收集 train2 全部标注
boxes = collections.defaultdict(list)   # cls -> [(w,h,area), ...]
imgs_with = collections.defaultdict(set)
tiny = collections.defaultdict(int)     # 极小框计数
degenerate = collections.defaultdict(int)  # 退化框(w或h≈0)
for f in glob.glob(ROOT + "/labels/train2/*.txt"):
    stem = os.path.basename(f)
    for line in open(f):
        p = line.split()
        if len(p) >= 5:
            c = int(float(p[0])); w = float(p[3]); h = float(p[4])
            boxes[c].append((w, h, w * h))
            imgs_with[c].add(stem)
            if w * h < 0.0005:
                tiny[c] += 1
            if w < 1e-3 or h < 1e-3:
                degenerate[c] += 1

print("类别  名称       实例数  覆盖图数  中位框面积%  极小框(<0.05%)  退化框  长宽比异常")
allc = sorted(boxes, key=lambda c: -len(boxes[c]))
for c in allc:
    bs = boxes[c]; n = len(bs)
    medA = statistics.median(a for _, _, a in bs) * 100
    ar = [max(w/h, h/w) if h > 0 and w > 0 else 99 for w, h, _ in bs]
    extreme_ar = sum(1 for x in ar if x > 10)
    flag = " <== 难类" if c in HARD else ""
    nm = HARD.get(c, "")
    print(f"  {c:<3} {nm:<8} {n:>5}   {len(imgs_with[c]):>5}    {medA:7.3f}     {tiny[c]:>5}({100*tiny[c]//max(n,1)}%)  {degenerate[c]:>4}   {extreme_ar:>4}{flag}")

print("\n难类标注小结:")
for c, nm in sorted(HARD.items()):
    if c in boxes:
        n = len(boxes[c]); imgs = len(imgs_with[c])
        per_img = n / max(imgs, 1)
        print(f"  cls{c}({nm}): {n}框/{imgs}图 = {per_img:.1f}框/图; 极小框{100*tiny[c]//max(n,1)}%; 退化{degenerate[c]}")
