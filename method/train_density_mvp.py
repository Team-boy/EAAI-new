"""cls18 密度分支训练."""
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from method.density_dataset import Cls18DensityDataset
from method.density_net import DensityNet

ROOT = "/home/fenghn/CvTest/data/Dataset B"


def weighted_bce(pred, tgt, pos_w=10.0):
    w = torch.ones_like(tgt) + (pos_w - 1.0) * tgt
    return nn.functional.binary_cross_entropy(pred, tgt, weight=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--hm", type=int, default=160)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="/home/fenghn/CvTest/method/runs/density_mvp.pt")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = f"cuda:{a.device}"
    dl = DataLoader(Cls18DensityDataset(ROOT, "train2", a.img, a.hm),
                    batch_size=a.bs, shuffle=True, num_workers=4, drop_last=True)
    net = DensityNet(a.hm).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    epochs = 1 if a.smoke else a.epochs
    net.train()
    for ep in range(epochs):
        tot = nb = 0
        for img, hm in dl:
            img, hm = img.to(dev), hm.to(dev)
            opt.zero_grad()
            loss = weighted_bce(net(img), hm)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if a.smoke and nb >= 3:
                break
        print(f"epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print("SAVED", a.out)


if __name__ == "__main__":
    main()
