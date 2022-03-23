from matplotlib.cbook import flatten
import torch as th
import torchvision as thv
import torchvision.transforms as transforms
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
import glob
import pandas as pd
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
    def __init__(self, o=0):
        super().__init__()
        self.o = o

    def forward(self, x):
        if self.o:
            return x.reshape(-1, self.o)
        else:
            return x.reshape(x.size(0), -1)

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

        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))

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


class fcnn(nn.Module):
    def __init__(self, dims, bn=False, bias=True):
        super(fcnn, self).__init__()

        self.dims = dims

        self.handles = []
        self.layers = nn.ModuleList([flatten_t(dims[0])])

        for i in range(len(dims)-2):
            l = nn.Linear(dims[i], dims[i+1], bias=bias)
            self.layers.append(l)
            l = nn.ReLU()
            if bn:
                self.layers.append(nn.BatchNorm1d(dims[i+1], affine=False))
            self.layers.append(l)
        self.layers.append(nn.Linear(dims[-2], dims[-1]))
        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


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

        print('Num parameters: ', sum([p.numel() for p in self.m.parameters()]))

    def forward(self, x):
        return self.m(x)


def get_data(name='CIFAR10', sub_sample=0, dev='cpu', resize=1, aug=False):
    assert name in ['CIFAR10', 'CIFAR100', 'MNIST']

    f = getattr(thv.datasets, name)

    if name in ['CIFAR10', 'CIFAR100']: 
        sz = 32//resize
        if aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]) 
        else:
            transform_train = transforms.ToTensor()
    elif name == 'STL10': sz = 96//resize
    elif name == 'MNIST': 
        sz = 28//resize
    else: assert False

    ds = {'train': f('../data', train=True, download=False, transform=transform_train),
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

def load_d(loc, cond={}, avg_err=False):
    r = []
    for f in glob.glob(os.path.join(loc, '*}.p')):
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if all(configs[k] == v for (k, v) in cond.items()):
            d = th.load(f)
            for i in range(len(d)):
                t = {}
                t.update(configs)
                t.update({'t': i})
                t.update(d[i])
                r.append(t)

    d = pd.DataFrame(r)
    if avg_err:
        d['err'] = d.apply(lambda r: r.e.mean().item(), axis=1)
        d['verr'] = d.apply(lambda r: r.ev.mean().item(), axis=1)
        d['favg'] = d.apply(lambda r: r.f.mean().item(), axis=1)
        d['vfavg'] = d.apply(lambda r: r.fv.mean().item(), axis=1)

    print(d.keys(), len(d))
    del r

    return d


def avg_model(d, groupby=['m', 't'], probs=False, avg=None, get_err=True, compute_distance=False, dev='cuda', distf=th.cdist):
    key = ['yh', 'yvh']
    n_data = d[key].iloc[0].shape[0]

    if avg is None:
        #         avg = {}
        for k in key:
            if isinstance(d.iloc[0][k], th.Tensor):
                if not probs:
                    d[k] = d.apply(lambda r: np.exp(
                        r[k].flatten().numpy()), axis=1)
                else:
                    d[k] = d.apply(lambda r: r[k].flatten().numpy(), axis=1)

        avg = d.groupby(groupby)[key].mean(numeric_only=False).reset_index()
#         print(avg)
    if get_err:
        for k in key:
            ykey = k.strip("h")
            y = get_data(dev='cuda')[ykey]
            n = len(y)
            preds = np.argmax(np.stack(avg[k]).reshape(-1, n, 10), -1)
            err = ((th.Tensor(preds).cuda() != y).sum(1) / n).cpu().numpy()
            avg[f'{ykey[1:]}err'] = err

    if compute_distance:
        dists = []
        for i in range(len(avg)):
            config = {}
            for k in groupby:
                v = avg.iloc[i][k]
                if isinstance(v, str):
                    v = f"'{v}'"
                config[k] = v
            ii = get_idx(d, "&".join(
                [f"{k} == {v}" for (k, v) in config.items()]))

            for k in key:
                x1 = avg.iloc[i][k].reshape(1, -1, 10)
                x2 = np.stack(d.loc[ii][k]).reshape(len(ii), -1, 10)
                x1 = th.Tensor(x1).transpose(0, 1).to(dev)
                x2 = th.Tensor(x2).transpose(0, 1).to(dev)
#                 dist = -th.log(th.bmm(x2, x1.transpose(1,2))).mean(0)
                dist = distf(x2, x1).mean(0)
                for (j, dj) in enumerate(dist):
                    dic = dict(dist=dj.item(), key=k)
                    for (kc, vc) in config.items():
                        if isinstance(vc, str):
                            vc = vc.strip("''")
                        dic.update({kc: vc})
#                     dic.update(config)
                    dists.append(dic)
        dists = pd.DataFrame(dists)
        return avg, dists
    return avg

# def avg_model(d, groupby=['m', 't'], get_err=True, probs=False, compute_distance=False, dev='cuda'):
#     def get_idx(dd, cond):
#         return dd.query(cond).index.tolist()

#     key = ['yh', 'yvh']
#     n_data = d[key].iloc[0].shape[0]

#     avg = {}
#     if not probs:
#         for k in key:
#             d[k] = d.apply(lambda r: np.exp(r[k].numpy()), axis=1)

#     avg = d.groupby(groupby)[key].mean(numeric_only=False).reset_index()

#     if get_err:
#         for k in key:
#             ykey = k.strip("h")
#             y = get_data(dev='cuda')[ykey]
#             n = len(y)
#             preds = np.argmax(np.stack(avg[k]).reshape(-1, n, 10), -1)
#             err = ((th.Tensor(preds).cuda() != y).sum(1) / n).cpu().numpy()
#             avg[f'{ykey[1:]}err'] = err

#     if compute_distance:
#         dists = []
#         for i in range(len(avg)):
#             config = {}
#             for k in groupby:
#                 v = avg.iloc[i][k]
#                 if isinstance(v, str):
#                     v = f"'{v}'"
#                 config[k] = v
#             ii = get_idx(d, "&".join(
#                 [f"{k} == {v}" for (k, v) in config.items()]))

#             x1 = avg.iloc[i][key].reshape(n, 1, -1)
#             x2 = th.stack(d.loc[ii][key]).reshape(n, len(ii), -1)
#             x1 = th.sqrt(th.Tensor(x1)).to(dev)
#             x2 = th.sqrt(th.Tensor(x2)).to(dev)

#             dist = -th.log(th.bmm(x2, x1.transpose(1, 2))).sum(0)

#             for (j, dj) in enumerate(dist):
#                 dic = dict(dist=dj.item())
#                 dic.update(config)
#                 dists.append(dic)
#         dists = pd.DataFrame(dists)
#         return avg, dists
#     return avg
