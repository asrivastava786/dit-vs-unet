import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetBlock(3, 64)
        self.down2 = UNetBlock(64, 128)

        self.mid = UNetBlock(128, 128)

        self.up1 = UNetBlock(128, 64)
        self.up2 = UNetBlock(64, 3)

    def forward(self, x, t=None):
        d1 = self.down1(x)
        d2 = self.down2(d1)

        mid = self.mid(d2)

        u1 = self.up1(mid)
        return self.up2(u1)
