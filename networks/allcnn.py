from networks.net_utils import * 

class allcnn_t(nn.Module):
    def __init__(self, num_classes=10, c1=96, c2=144, bn=False):
        super().__init__()
        d = 0.5

        def convbn(ci, co, ksz, s=1, pz=0):
            layers = [
                nn.Conv2d(ci, co, ksz, stride=s, padding=pz),
                nn.ReLU(True),
            ]
            if bn:
                layers.append(nn.BatchNorm2d(co, affine=True))
            return nn.Sequential(*layers)

        self.m = nn.Sequential(
            convbn(3, c1, 3, 1, 1),
            convbn(c1, c1, 3, 2, 1),
            convbn(c1, c2, 3, 1, 1),
            convbn(c2, c2, 3, 2, 1),
            convbn(c2, num_classes, 1, 1),
            nn.AvgPool2d(8),
            flatten_t(num_classes))

        print('Num parameters: ', sum([p.numel()
              for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)
