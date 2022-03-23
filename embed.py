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

def get_idx(dd, cond):
    return dd.query(cond).index.tolist()

def drop_untrained(dd, key='err', th=0.01, verbose=False):
    tmax = dd['t'].max()
    ii = get_idx(dd, f"t == {tmax} & {key} > {th}")
    iis = [j for i in ii for j in range(i-tmax, i+1, 1)]
    if verbose:
        print(len(ii))
        print(dd[['m', 'opt', 'bn', 'seed']].iloc[ii])
    return dd.drop(iis)

def embed(dd, fn='', ss=slice(0,-1,1), probs=False, ne=3, key='yh', force=False, idx=None, dev='cuda', distf='bhat', reduction='sum', loc='inpca_results'):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dc = dd[idx]
    th.save(dc, os.path.join(loc, 'didx_%s.p' % fn))
    n = len(dd) # number of models
    x = th.Tensor(np.stack([dd.iloc[i][key][ss] for i in range(n)])).to(dev)
    print(x.shape)

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        w = dist_(x, probs=probs, dev=dev, distf=distf, reduction=reduction)
        print('Saving w')
        th.save(w, os.path.join(loc, 'w_%s.p' % fn))
    else:
        print('Found: ', os.path.join(loc, 'w_%s.p' % fn))
        w = th.load(os.path.join(loc, 'w_%s.p' % fn))

    l = np.eye(w.shape[0]) - 1.0/w.shape[0]
    w = -l @ w @ l / 2
    r = proj_(w, n, ne)
    th.save(r, os.path.join(loc, 'r_%s.p' % fn))
    return

def dist_(xs, probs=True, dev='cuda', distf='bhat', reduction='sum'):
    if not isinstance(xs, th.Tensor):
        xs = th.Tensor(xs)
    xs = xs.to(dev)
    n = len(xs)
    xs = th.moveaxis(xs, 0, 1)
    w = np.zeros([n, n])
    nc = 200 if n < 3000 else 500
    print('chunks: ', nc)
    if distf == 'bhat':
        xs = xs if probs else th.exp(xs)

    for aa in tqdm.tqdm(th.chunk(xs, nc)):
        if distf == 'bhat':
            aa = th.sqrt(aa)
            w_ = -th.log(th.bmm(aa, aa.transpose(1, 2)))
            w_[w_ == th.inf] = 100
            w_[w_ < 0] = 0
            w += getattr(w_, reduction)(0).cpu().numpy()
        elif distf == 'kl':
            # import ipdb;ipdb.set_trace()
            if probs:
                logaa = th.log(aa)
                logaa[logaa==-th.inf] = 100
                w_ = (aa * logaa).sum(-1, keepdim=True) - th.bmm(aa, logaa.transpose(1, 2))
            else:
                w_ = (th.exp(aa) * aa).sum(-1, keepdim=True) - th.bmm(th.exp(aa), (aa).transpose(1, 2))
            w += getattr(w_, reduction)(0).cpu().numpy()
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

    # idx = ['seed', 'm', 'opt', 't', 'err', 'verr', 'favg', 'vfavg', 'bn']
    # d = d[idx]
    # for k in ['yh', 'yvh']:
        # d[k] = d.apply(lambda r: np.exp(r[k].numpy()), axis=1)
    # d.to_pickle(os.path.join(loc, "all_models.pkl"))
    # d = drop_untrained(d)
    # avg = avg_model(d, get_err=True, probs=True, compute_distance=False,
                    # groupby=['bn', 'm', 't', 'opt'])
    # avg.to_pickle(os.path.join(loc, "avg_models_clean.pkl") ) 
    # idx = ['m', 'opt', 't', 'err', 'verr', 'bn']
    # for key in ['yh', 'yvh']:
    #     fn = f'{key}_new_avg'
    #     embed(avg, fn=fn, probs=True, ss=slice(0,-1,2), key=key, idx=idx, force=False, dev=dev)

    # d = pd.read_pickle(os.path.join(loc, "all_models.pkl"))
    # d['avg'] = False
    # d_avg = pd.read_pickle(os.path.join(loc, "avg_models_clean.pkl"))
    # avg['avg'] = True
    # d = pd.concat([d, avg], axis=0)

    # data = get_data()
    # import torch.nn.functional as F
    # true = pd.Series(dict(m='true', yh=F.one_hot(data['y']), yvh=F.one_hot(data['yv'])))
    # d = d.append(true, ignore_index=True)
    import ipdb; ipdb.set_trace()

    d = load_d(loc, cond={'bs': 200, 'aug': True, 'wd': 0.0}, avg_err=False)
    end_models = d.iloc[get_idx(d, f"t == {d['t'].max()}")]
    enderrs = np.stack(end_models['e'].values).sum(0)
    endverrs = np.stack(end_models['ev'].values).sum(0)
    iis = np.where(enderrs>max(enderrs)//2)[0]
    ivs = np.where(endverrs>max(endverrs)//2)[0]
    i = iis[0]
    iv = ivs[0]
    d['iserr'] = np.stack(d['e'].values)[:, i]
    d['viserr'] = np.stack(d['ev'].values)[:, iv]
    idx = ['seed', 'm', 'opt', 't', 'bn', 'iserr', 'viserr']
    for key in ['yh', 'yvh']:
        fn = f'{key}_new_subset_{i}_{iv}'
        # idxs = th.load(os.path.join(loc, f'{key}_idx.p'))
        ss = i if key == 'yh' else iv
        embed(d, fn=fn, ss=[ss], probs=False, key=key, idx=idx, force=True, distf='bhat')

if __name__ == '__main__':
    main()