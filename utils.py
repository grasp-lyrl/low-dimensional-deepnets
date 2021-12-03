import torch as th
import torchvision as thv
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn

from torch.cuda.amp import autocast, GradScaler 

jacobian = autograd.functional.jacobian
hessian = autograd.functional.hessian

import os, sys, pdb, tqdm, random, json, gzip, bz2
from copy import deepcopy
from collections import defaultdict
from functools import partial

def setup(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

class flatten_t(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class wide_resnet_t(nn.Module):

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, dropout_rate, stride=1, bn=True):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes, affine=False) if bn else nn.Identity()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.bn2 = nn.BatchNorm2d(planes, affine=False) if bn else nn.Identity()
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                )

        def forward(self, x):
            out = self.dropout(self.conv1(F.relu(self.bn1(x))))
            out = self.conv2(F.relu(self.bn2(out)))
            out = out + self.shortcut(x)
            return out

    def __init__(self, depth, widen_factor, dropout_rate=0.,
                 num_classes=10, in_planes=16, bn=True):
        super().__init__()
        self.in_planes = in_planes

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

#         print('wide_resnet_t %d-%d-%d' %(depth, k, in_planes))
        nStages = [in_planes, in_planes*k, 2*in_planes*k, 4*in_planes*k]

        self.conv1 = self.conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(self.wide_basic, nStages[1], n, dropout_rate, stride=1, bn=bn)
        self.layer2 = self._wide_layer(self.wide_basic, nStages[2], n, dropout_rate, stride=2, bn=bn)
        self.layer3 = self._wide_layer(self.wide_basic, nStages[3], n, dropout_rate, stride=2, bn=bn)
        self.bn1 = nn.BatchNorm2d(nStages[3], affine=False) if bn else nn.Identity()
        self.view = flatten_t()
        self.linear = nn.Linear(nStages[3], num_classes)

        self.activations = defaultdict(list)
        self.handles = []

    def add_activation_hooks(self):
        def save_activation(m, inp, op, name):
            self.activations[name].append(op.detach())
        hooks = hooks = [partial(save_activation, name=f'layer{i}') for i in range(1, 5)]
        self.handles.append(self.layer1.register_forward_hook(hooks[0]))
        self.handles.append(self.layer2.register_forward_hook(hooks[1]))
        self.handles.append(self.layer3.register_forward_hook(hooks[2]))
        self.handles.append(self.view.register_forward_hook(hooks[3]))

    def remove_activation_hooks(self):
        self.activations = defaultdict(list)
        for h in self.handles:
            h.remove()
        self.handles = []

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bn):
        strides = [stride] + [1]*max(int(num_blocks)-1, 0)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, bn))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out,1)
        out = self.view(out)
        out = self.linear(out)
        return out

def get_data(name='CIFAR10', sub_sample=0, dev='cpu', resize=1):
    assert name in ['CIFAR10', 'CIFAR100', 'MNIST']

    f = getattr(thv.datasets, name)
    if name in ['CIFAR10', 'CIFAR100']: sz = 32//resize
    elif name == 'STL10': sz = 96//resize
    elif name == 'MNIST': sz = 28//resize
    else: assert False

    ds = {'train': f('../data', train=True, download=False),
          'val': f('../data', train=False, download=False)}

    x,y = th.tensor(ds['train'].data).float(), th.tensor(ds['train'].targets).long()
    xv,yv = th.tensor(ds['val'].data).float(), th.tensor(ds['val'].targets).long()

    if name == 'MNIST':
        x, xv = x[:,:,:,None], xv[:,:,:,None]

    # preprocess to make it zero mean and unit variance
    x, xv = th.transpose(x, 1, 3), th.transpose(xv, 1, 3)
    x = (x-th.mean(x, dim=[0,2,3], keepdim=True))/th.std(x, dim=[0,2,3], keepdim=True)
    xv = (xv-th.mean(xv, dim=[0,2,3], keepdim=True))/th.std(xv, dim=[0,2,3], keepdim=True)
    x = F.interpolate(x, size=sz, mode='bicubic', align_corners=False)

    # subsample the dataset, make sure it is balanced
    if sub_sample > 0:
        ss = int(len(x)*sub_sample//10)
        y, ii = th.sort(y)
        x = x[ii]
        xs, ys = th.chunk(x, 10), th.chunk(y, 10)
        x = th.cat([xx[:ss] for xx in xs])
        y = th.cat([yy[:ss] for yy in ys])

    ds = dict(x=x,y=y,xv=xv,yv=yv)
    for k,v in ds.items():
        if dev == 'cpu' and th.cuda.is_available():
            ds[k] = ds[k].pin_memory()
        else:
            ds[k] = ds[k].to(dev)
    return ds

def relabel_data(fn, y, frac=0.1, dev='cuda'):
    d = th.load(fn)
    yh = d[-1]['yh'].to(dev)
    _, yi = th.sort(yh, dim=1)
    y_new = yi[:, -2]
    # balanced relabel
    ss = int(len(yh)*frac//10)
    ys = th.chunk(th.arange(len(y)), 10)
    for yy in ys:
        idx = yy[th.randperm(len(yy))][:ss]
        y[idx] = y_new[idx]