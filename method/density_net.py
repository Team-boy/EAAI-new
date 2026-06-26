"""轻量密度分割网络: ResNet18 编码器(到 layer2) + 上采样头."""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class DensityNet(nn.Module):
    def __init__(self, hm_size=160):
        super().__init__()
        self.hm_size = hm_size
        bb = resnet18(weights=None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        f = self.stem(x)
        y = self.head(f)
        y = nn.functional.interpolate(y, size=(self.hm_size, self.hm_size),
                                      mode="bilinear", align_corners=False)
        return torch.sigmoid(y)
