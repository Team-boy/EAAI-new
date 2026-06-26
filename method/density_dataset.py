"""cls18 密度分支数据集: 图像 + box弱监督密度GT."""
import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from method.density_gt import boxes_to_heatmap

CLS18 = 18


class Cls18DensityDataset(Dataset):
    def __init__(self, root, split="train2", img_size=640, hm_size=160, sigma=2.0):
        self.img_dir = f"{root}/images/{split}"
        self.lbl_dir = f"{root}/labels/{split}"
        self.img_size, self.hm_size, self.sigma = img_size, hm_size, sigma
        exts = ("*.jpg", "*.JPG", "*.png", "*.jpeg", "*.bmp")
        self.imgs = sorted(p for e in exts for p in glob.glob(f"{self.img_dir}/{e}"))

    def __len__(self):
        return len(self.imgs)

    def _boxes(self, stem):
        lbl = f"{self.lbl_dir}/{stem}.txt"
        out = []
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.split()
                if len(p) >= 5 and int(float(p[0])) == CLS18:
                    out.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        return out

    def __getitem__(self, i):
        path = self.imgs[i]
        stem = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        ten = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).contiguous()
        hm = boxes_to_heatmap(self._boxes(stem), self.hm_size, self.hm_size, self.sigma)
        return ten, torch.from_numpy(hm).unsqueeze(0)
