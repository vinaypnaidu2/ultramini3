import torch.nn.functional as F
import torch
import torch.nn as nn 
from unet_parts import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.down3 = Down(192, 384)
        factor = 2 if bilinear else 1
        self.down4 = Down(384, 768 // factor)
        self.up1 = Up(768, 384 // factor, bilinear)
        self.up2 = Up(384, 192 // factor, bilinear)
        self.up3 = Up(192, 96 // factor, bilinear)
        self.up4 = Up(96, 48, bilinear)
        self.pre = nn.Conv2d(48, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))
        return x
