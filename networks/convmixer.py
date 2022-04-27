from networks.net_utils import *

# https://openreview.net/forum?id=TVHS5Y4dNvM
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class convmixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=10, bn=False):
        super().__init__()
        self.m = [
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        ]
        if bn:
            self.m.append(nn.BatchNorm2d(dim))
        for i in range(depth):
            block = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size,
                              groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                ) if bn else nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size,
                              groups=dim, padding="same"),
                    nn.GELU(),
                )
            self.m.append(Residual(block))
            self.m.append(nn.Conv2d(dim, dim, kernel_size=1))
            self.m.append(nn.GELU())
            if bn:
                self.m.append(nn.BatchNorm2d(dim))
        self.m += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(dim, n_classes)]
        self.m = nn.Sequential(*self.m)

    def forward(self, x):
        return self.m(x)
