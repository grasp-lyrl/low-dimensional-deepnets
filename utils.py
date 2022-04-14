import torch as th
import torchvision as thv
import torchvision.transforms as transforms
import numpy as np
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
from scipy.interpolate import interpn


def setup(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def get_idx(dd, cond):
    return dd.query(cond).index.tolist()


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


def load_d(loc, cond={}, avg_err=False, numpy=True, probs=False, drop=True):
    r = []
    for f in glob.glob(os.path.join(loc, '*}.p')):
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if all(configs[k] in v for (k, v) in cond.items()):
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

    for k in ['yh', 'yvh']:
        if probs:
            d[k] = d.apply(lambda r: th.exp(r[k]), axis=1)
        if numpy:
            d[k] = d.apply(lambda r: r[k].numpy(), axis=1) 

    if drop:
        d = drop_untrained(d, key='err', th=0.01, verbose=False).reset_index()

    print(d.keys(), len(d))
    del r

    return d


def drop_untrained(dd, key='err', th=0.01, verbose=False):
    tmax = dd['t'].max()
    ii = get_idx(dd, f"t == {tmax} & {key} > {th}")
    iis = [j for i in ii for j in range(i-tmax, i+1, 1)]
    if verbose:
        print(len(ii))
        print(dd[['m', 'opt', 'bn', 'seed']].iloc[ii])
    return dd.drop(iis)


def avg_model(d, groupby=['m', 't'], probs=False, avg=None, get_err=True, update_d=False,
              compute_distance=False, dev='cuda', distf=lambda x, y: th.cdist(x, y).mean(0)):
    keys = ['yh', 'yvh']
    d_return = {}

    n_data = {k: d[k].iloc[0].shape[0] for k in keys}
    if avg is None:
        for k in keys:
            if isinstance(d.iloc[0][k], th.Tensor):
                d[k] = d.apply(lambda r: r[k].flatten().numpy(), axis=1)
            else:
                d[k] = d.apply(lambda r: r[k].flatten(), axis=1)
            if not probs:
                d[k] = d.apply(lambda r: np.exp(r[k]), axis=1)
                   
        avg = d.groupby(groupby)[keys].mean(numeric_only=False).reset_index()
        for k in keys:
            avg[k] = avg.apply(lambda r: r[k].reshape(n_data[k], -1), axis=1)
            d[k] = d.apply(lambda r: r[k].reshape(n_data[k], -1), axis=1)

    if get_err:
        for k in keys:
            ykey = k.strip("h")
            y = get_data(dev='cuda')[ykey]
            n = len(y)
            out = np.stack(avg[k])
            preds = np.argmax(out, -1)
            err = ((th.Tensor(preds).cuda() != y).sum(1) / n).cpu().numpy()
            avg[f'{ykey[1:]}err'] = err


    if compute_distance:
        dists = []
        indices = d.groupby(groupby).indices
        for i in range(len(avg)):
            config = tuple(avg.iloc[i][groupby])
            ii = indices[config]
            for k in keys:
                x1 = th.Tensor(avg.iloc[i][k]).unsqueeze(0).transpose(0, 1).to(dev)
                x2 = th.Tensor(np.stack(d.iloc[ii][k])).transpose(0, 1).to(dev)
                dist = distf(x2, x1)
                for (j, dj) in enumerate(dist):
                    dic = dict(dist=dj.item(), key=k)
                    dic.update({groupby[i]:config[i] for i in range(len(groupby))})
                    dists.append(dic)
        dists = pd.DataFrame(dists)
        d_return['dists'] = dists

    if update_d:
        avg['avg'] = True
        avg['seed'] = -1
        d['avg'] = False
        d = pd.concat([avg, d]).reset_index()
        d_return['d'] = d
    else:
        d_return['avg'] = avg

    return d_return


def interpolate(d, ts, pts, columns=['seed', 'm', 'opt', 'avg'], keys=['yh', 'yvh'], dev='cuda'):
    r = []
    N = len(pts)
    y = get_data(dev=dev)
    configs = d.groupby(columns).indices
    for (c, idx) in configs.items():
        traj_interp = {}
        for k in keys:
            traj = np.stack(d.iloc[idx][k])
            traj_ = interpn(ts.reshape(1, -1), traj, pts)
            traj_interp[k] = th.Tensor(traj_).to(dev)
        for i in range(N):
            t = {}
            t.update({columns[i]: ci for (i, ci) in enumerate(c)})
            t.update({'t': pts[i]})
            for k in keys:
                ti = traj_interp[k][i, :].reshape(-1, 10)
                t.update({k: ti.cpu().numpy()})
                f = -th.gather(th.log(ti+1e-8), 1, y[k[:-1]].view(-1, 1)).mean()
                acc = th.eq(th.argmax(ti, axis=1), y[k[:-1]]).float()
                if k == 'yh':
                    t.update({'favg': f.mean().item(), 'err': 1-acc.mean().item()})
                else:
                    t.update({'vfavg': f.mean().item(),
                            'verr': 1-acc.mean().item()})
            r.append(t)
    d = pd.DataFrame(r)
    return d
