import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleDiT(nn.Module):
    def __init__(self, dim=256, depth=6):
        super().__init__()
        self.patch = PatchEmbed()

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                batch_first=True
            ) for _ in range(depth)
        ])

        self.head = nn.Linear(dim, dim)

    def forward(self, x, t=None):
        x = self.patch(x)

        for blk in self.blocks:
            x = blk(x)

        return self.head(x)
