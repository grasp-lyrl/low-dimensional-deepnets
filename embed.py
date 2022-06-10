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
import distance

def embed(dd, extra_pts=None, fn='', ss=slice(0,None,1), probs=False, ne=3, key='yh', force=False, idx=None, dev='cuda', distf='dbhat', 
          reduction='sum', loc='inpca_results', chunks=1):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dc = dd[idx]
    if extra_pts is not None:
        qc = extra_pts.loc[:, extra_pts.columns.isin(idx)] 
        dc = pd.concat([dc, qc])
        q = th.Tensor(np.stack([extra_pts.iloc[i][key][ss] for i in range(len(extra_pts))])).cpu()
    th.save(dc, os.path.join(loc, 'didx_%s.p' % fn))
    n = len(dd) # number of models

    x = th.Tensor(np.stack([dd.iloc[i][key][ss] for i in range(n)])).cpu()

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        if 'kl' in distf:
            w = getattr(distance, distf)(x, x, reduction=reduction, dev=dev, chunks=chunks, probs=probs)
        else:
            x = th.exp(x) if not probs else x
            w = getattr(distance, distf)(x, x, reduction=reduction, dev=dev, chunks=chunks)
        # w = dist_(x, probs=probs, dev=dev, distf=distf, reduction=reduction)
        print('Saving w')
        th.save(w, os.path.join(loc, 'w_%s.p' % fn))
    else:
        print('Found: ', os.path.join(loc, 'w_%s.p' % fn))
        w = th.load(os.path.join(loc, 'w_%s.p' % fn))

    d_mean = w.mean(0)
    l = np.eye(w.shape[0]) - 1.0/w.shape[0]
    w = -l @ w @ l / 2 
    r = proj_(w, n, ne)
    if extra_pts is not None:
        q = lazy_embed(q, x, w, d_mean, evals=r['e'], evecs=r['v'], distf=distf, ne=ne)
        r['xp'] = np.vstack([r['xp'], q]) 
    th.save(r, os.path.join(loc, 'r_%s.p' % fn))
    return


# Calculate the embedding of a distibution q in the intensive embedding of models p_list with divergence=distance, supply d_list the precalculated matrix of distances pf p_list.
def lazy_embed(q, ps, w, d_mean, evals=None, evecs=None, distf='dbhat', ne=3, chunks=1):
    # w: centered pairwise distance, d_mean: mean before centering
    N, _, _ = ps.shape
    dp = getattr(distance, distf)(q, ps, chunks=chunks)
    d_mean_mean = np.mean(d_mean)
    if (evals is not None) or (evecs is not None):
        _, _, evals, evecs = proj_(w, N, ne).values() 
    dp_mean = dp-np.mean(dp)-d_mean+d_mean_mean
    dp_mean = -.5*dp_mean  
    sqrtsigma = np.sqrt(np.abs(evals))
    return ((1/sqrtsigma)*np.matmul(dp_mean, evecs))


def dist_(xs, probs=True, dev='cuda', distf='bhat', reduction='sum', chunks=200):
    # compute pairwise distance between all elements of xs
    if not isinstance(xs, th.Tensor):
        xs = th.Tensor(xs)
    # xs = xs.to(dev)
    n = len(xs)
    xs = th.moveaxis(xs, 0, 1)
    w = np.zeros([n, n])
    nc = chunks
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
        w = w/xs.size(0)
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
    loc = 'results/plots'

    # d = pd.read_pickle(os.path.join(loc, "all_models.pkl"))
    # d['avg'] = False
    # d_avg = pd.read_pickle(os.path.join(loc, "avg_models_clean.pkl"))
    # avg['avg'] = True
    # d = pd.concat([d, avg], axis=0)

    # data = get_data()
    # import torch.nn.functional as F
    # true = pd.Series(dict(m='true', yh=F.one_hot(data['y']), yvh=F.one_hot(data['yv'])))
    # d = d.append(true, ignore_index=True)
    # models = ["wr-10-4-8"]
    # opts = ["Adam", "SGD"]
    models = ["wr-4-8", "allcnn-96-144", "fc-1024-512-256-128"]
    opts = ["adam", "sgd", "sgdn"]
    # d = th.load(os.path.join(loc, 'new_avg.p'))
    # d['yh'] = d.apply(lambda r: r.yh.squeeze(), axis=1)
    # d = avg_model(d, groupby=['m', 'opt', 't'], probs=True, get_err=True,
    #                 update_d=True, compute_distance=False, dev='cuda', keys=['yh'])['d']
    d = load_d(loc, cond={'aug': ["na", True], 'm': models, 'opt':opts, 
                'corner':["na", "uniform", "subsample-200", "subsample-2000"]},
            avg_err=True, drop=0.0, probs=True)
    # T = 45000
    # ts = []
    # for t in range(T):
    #     if t < T//10:
    #         if t % (T//100) == 0:
    #             ts.append(t)
    #     else:
    #         if t % (T//10) == 0 or (t == T-1):
    #             ts.append(t)
    # pts = np.concatenate(
    #     [np.arange(ts[i], ts[i+1], (ts[i+1]-ts[i]) // 5) for i in range(len(ts)-1)])
    # ts = np.expand_dims(np.array(ts), 0)
    # d = avg_model(d, groupby=['m', 'opt', 't'], probs=True, get_err=True,
    #                   update_d=True, compute_distance=False, dev='cpu')['d']
    # d = interpolate(d, ts, pts, columns=['seed', 'm', 'opt', 'avg'], keys=[
    #                 'yh', 'yvh'], dev='cpu')
    # d = interpolate(d, ts, pts, columns=['seed', 'm', 'opt'], keys=[
    #                 'yh', 'yvh'], dev='cpu')
    
    # for key in ['yh', 'yvh']:
    data = get_data()
    y, yv = th.tensor(data['train'].targets).long(), th.tensor(data['val'].targets).long()
    y_ = np.zeros((len(y), y.max()+1))
    y_[np.arange(len(y)), y] = 1
    yv_ = np.zeros((len(yv), yv.max()+1))
    yv_[np.arange(len(yv)), yv] = 1
    extra_pts = [dict(seed=0, m='true', t=th.inf, err=0., verr=0., yh=y_, yvh=yv_), 
                dict(seed=0, m='random', t=0, err=0.9, verr=0.9, 
                yh=np.ones([len(y), y.max()+1])/(y.max()+1), 
                yvh=np.ones([len(yv), yv.max()+1])/(yv.max()+1))
                ]
    extra_pts = pd.DataFrame(extra_pts)
    for key in ['yh', 'yvh']:
        # fn = f'{key}_new_subset_{i}_{iv}'
        # idxs = th.load(os.path.join(loc, f'{key}_idx.p'))
        # ss = i if key == 'yh' else iv
        fn = f'{key}_plot_all'
        idx = ['seed', 'm', 'opt', 't', 'err', 'favg', 'bs', 'aug', 'bn', 'corner']
        print(d['seed'].unique())
        embed(d, extra_pts=extra_pts, fn=fn, ss=slice(0, -1, 2), probs=True, key=key,
              idx=idx, force=True, distf='dbhat', reduction='mean', chunks=800)


if __name__ == '__main__':
    main()
