from networks.net_utils import *

class wide_resnet(nn.Module):

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, dropout, stride=1, bn=True):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(
                in_planes, affine=False) if bn else nn.Identity()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, padding=1, bias=False)
            self.dropout = nn.Dropout(p=dropout)
            self.bn2 = nn.BatchNorm2d(
                planes, affine=False) if bn else nn.Identity()
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1,
                              stride=stride, bias=True),
                )

        def forward(self, x):
            out = self.dropout(self.conv1(F.relu(self.bn1(x))))
            out = self.conv2(F.relu(self.bn2(out)))
            out = out + self.shortcut(x)
            return out

    def __init__(self, depth, widen_factor, dropout=0.,
                 num_classes=10, in_planes=16, bn=True):
        super().__init__()
        self.in_planes = in_planes

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

#         print('wide_resnet_t %d-%d-%d' %(depth, k, in_planes))
        nStages = [in_planes, in_planes*k, 2*in_planes*k, 4*in_planes*k]

        self.conv1 = self.conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            self.wide_basic, nStages[1], n, dropout, stride=1, bn=bn)
        self.layer2 = self._wide_layer(
            self.wide_basic, nStages[2], n, dropout, stride=2, bn=bn)
        self.layer3 = self._wide_layer(
            self.wide_basic, nStages[3], n, dropout, stride=2, bn=bn)
        self.bn1 = nn.BatchNorm2d(
            nStages[3], affine=False) if bn else nn.Identity()
        self.view = flatten_t()
        self.linear = nn.Linear(nStages[3], num_classes)

        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))

    def _wide_layer(self, block, planes, num_blocks, dropout, stride, bn):
        strides = [stride] + [1]*max(int(num_blocks)-1, 0)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes,
                          dropout, stride, bn))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.view(out)
        out = self.linear(out)
        return out
