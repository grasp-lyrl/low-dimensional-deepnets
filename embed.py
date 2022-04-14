import numpy as np
import scipy.linalg as sp
import torch as th

import os
import pdb
import sys
import json
import glob
import tqdm
import pandas as pd

from utils import *

def embed(dd, fn='', ss=slice(0,-1,1), probs=False, ne=3, key='yh', force=False, idx=None, dev='cuda', distf='bhat', 
          reduction='sum', loc='inpca_results'):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dc = dd[idx]
    th.save(dc, os.path.join(loc, 'didx_%s.p' % fn))
    n = len(dd) # number of models

    x = th.Tensor(np.stack([dd.iloc[i][key][ss] for i in range(n)])).cpu()

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        w = dist_(x, probs=probs, dev=dev, distf=distf, reduction=reduction)
        print('Saving w')
        th.save(w, os.path.join(loc, 'w_%s.p' % fn))
    else:
        print('Found: ', os.path.join(loc, 'w_%s.p' % fn))
        w = th.load(os.path.join(loc, 'w_%s.p' % fn))

    l = np.eye(w.shape[0]) - 1.0/w.shape[0]
    w = -l @ w @ l 
    r = proj_(w, n, ne)
    th.save(r, os.path.join(loc, 'r_%s.p' % fn))
    return

def dist_(xs, probs=True, dev='cuda', distf='bhat', reduction='sum'):
    if not isinstance(xs, th.Tensor):
        xs = th.Tensor(xs)
    # xs = xs.to(dev)
    n = len(xs)
    xs = th.moveaxis(xs, 0, 1)
    w = np.zeros([n, n])
    nc = 200 if n < 3000 else 800
    print('chunks: ', nc)
    if distf == 'bhat':
        xs = xs if probs else th.exp(xs)
    for aa in tqdm.tqdm(th.chunk(xs, nc)):
        aa = aa.to(dev)
        if distf == 'bhat':
            aa = th.sqrt(aa)
            w_ = -th.log(th.bmm(aa, aa.transpose(1, 2)))
            w_[w_ == th.inf] = 100
            w_[w_ < 0] = 0
            w += w_.sum(0).cpu().numpy()
        elif distf == 'kl':
            if probs:
                logaa = th.log(aa)
                logaa[logaa==-th.inf] = 100
                w_ = (aa * logaa).sum(-1, keepdim=True) - th.bmm(aa, logaa.transpose(1, 2))
            else:
                w_ = (th.exp(aa) * aa).sum(-1, keepdim=True) - th.bmm(th.exp(aa), (aa).transpose(1, 2))
            w += getattr(w_, reduction)(0).cpu().numpy()
        elif distf == 'skl':
            if probs:
                logaa = th.log(aa)
                logaa[logaa==-th.inf] = 100
                w1 = (aa * logaa).sum(-1, keepdim=True) - th.bmm(aa, logaa.transpose(1, 2))
                w2 = (aa * logaa).sum(-1, keepdim=True) - th.bmm(logaa, aa.transpose(1, 2))
                w_ = 0.5*(w1+w2)
            else:
                w1 = (th.exp(aa) * aa).sum(-1, keepdim=True) - th.bmm(th.exp(aa), (aa).transpose(1, 2))
                w2 = (th.exp(aa) * aa).sum(-1, keepdim=True) - th.bmm(aa, th.exp(aa).transpose(1, 2))
                w_ = 0.5*(w1+w2)
    if reduction == 'mean':
        w = w/n
    return w

def proj_(w, n, ne):
    print('Projecting')
    e1, v1 = sp.eigh(w, driver='evx', check_finite=False,
                     subset_by_index=[n-(ne+1), n-1])
    e2, v2 = sp.eigh(w, driver='evx', check_finite=False,
                     subset_by_index=[0, (ne+1)])
    e = np.concatenate((e1, e2))
    v = np.concatenate((v1, v2), axis=1)

    ii = np.argsort(np.abs(e))[::-1]
    e, v = e[ii], v[:, ii]
    xp = v*np.sqrt(np.abs(e))
    return dict(xp=xp, w=w, e=e, v=v)


def main():
    dev = 'cuda'
    loc = 'results/models/new'

    # d = pd.read_pickle(os.path.join(loc, "all_models.pkl"))
    # d['avg'] = False
    # d_avg = pd.read_pickle(os.path.join(loc, "avg_models_clean.pkl"))
    # avg['avg'] = True
    # d = pd.concat([d, avg], axis=0)

    # data = get_data()
    # import torch.nn.functional as F
    # true = pd.Series(dict(m='true', yh=F.one_hot(data['y']), yvh=F.one_hot(data['yv'])))
    # d = d.append(true, ignore_index=True)
    models = ["wr-4-8", "allcnn-96-144", "fc-1024-512-256-128"]
    opts = ["adam", "sgdn", "sgd"]
    loc = 'results/models/new'
    d = load_d(loc, cond={'bs': [200], 'aug': [True], 'wd': [0.0], 'bn': [True], 'm': models, 'opt': opts},
            avg_err=True, drop=True, probs=True)
    T = 45000
    ts = []
    for t in range(T):
        if t < T//10:
            if t % (T//100) == 0:
                ts.append(t)
        else:
            if t % (T//10) == 0 or (t == T-1):
                ts.append(t)
    pts = np.concatenate(
        [np.arange(ts[i], ts[i+1], (ts[i+1]-ts[i]) // 5) for i in range(len(ts)-1)])
    ts = np.expand_dims(np.array(ts), 0)
    d = avg_model(d, groupby=['m', 'opt', 't'], probs=True, get_err=True,
                      update_d=True, compute_distance=False, dev='cpu')['d']
    d = interpolate(d, ts, pts, keys=['yh', 'yvh'], dev='cpu')
    
    for key in ['yh', 'yvh']:
        # fn = f'{key}_new_subset_{i}_{iv}'
        # idxs = th.load(os.path.join(loc, f'{key}_idx.p'))
        # ss = i if key == 'yh' else iv
        fn = f'{key}_new_interpolate_with_avg'
        idx = ['seed', 'm', 'opt', 't', 'err', 'verr', 'favg', 'vfavg', 'avg']
        embed(d, fn=fn, ss=slice(0, -1, 4), probs=True, key=key,
              idx=idx, force=True, distf='bhat', reduction='mean')


if __name__ == '__main__':
    main()
