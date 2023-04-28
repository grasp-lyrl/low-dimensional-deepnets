import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from functools import partial


class flatten_t(nn.Module):
    def __init__(self, o=0):
        super().__init__()
        self.o = o

    def forward(self, x):
        if self.o:
            return x.reshape(-1, self.o)
        else:
            return x.reshape(x.size(0), -1)