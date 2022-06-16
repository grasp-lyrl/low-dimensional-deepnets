from networks.net_utils import *


class fcnn(nn.Module):
    def __init__(self, dims, bn=False, bias=True, dropout_rate=0.0):
        super(fcnn, self).__init__()

        self.dims = dims

        self.handles = []
        self.layers = nn.ModuleList([flatten_t(dims[0])])

        for i in range(len(dims)-2):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            self.layers.append(nn.ReLU())
            if bn:
                self.layers.append(nn.BatchNorm1d(dims[i+1], affine=False))
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(dims[-2], dims[-1]))
        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
