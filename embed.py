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


def embed(dd, extra_pts=None, fn='', ss=slice(0, None, 1), probs=False, ne=3, key='yh', force=False, idx=None, dev='cuda', distf='dbhat',
          reduction='sum', loc='inpca_results', chunks=1):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dc = dd[idx]
    if extra_pts is not None:
        qc = extra_pts.loc[:, extra_pts.columns.isin(idx)]
        dc = pd.concat([dc, qc])
        q = th.Tensor(np.stack([extra_pts.iloc[i][key][ss]
                      for i in range(len(extra_pts))])).cpu()
    th.save(dc, os.path.join(loc, 'didx_%s.p' % fn))
    n = len(dd)  # number of models

    x = th.Tensor(np.stack([dd.iloc[i][key][ss] for i in range(n)])).cpu()

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        if 'kl' in distf:
            w = getattr(distance, distf)(x, x, reduction=reduction,
                                         dev=dev, chunks=chunks, probs=probs)
        else:
            x = th.exp(x) if not probs else x
            w = getattr(distance, distf)(
                x, x, reduction=reduction, dev=dev, chunks=chunks)
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
        q = lazy_embed(
            q, x, w, d_mean, evals=r['e'], evecs=r['v'], distf=distf, ne=ne, chunks=chunks)
        r['xp'] = np.vstack([r['xp'], q])
    th.save(r, os.path.join(loc, 'r_%s.p' % fn))
    return


def xembed(d1, d2, extra_pts=None, fn='', ss=slice(0, None, 1), probs=False, ne=3, key='yh', force=False, idx=None, dev='cuda', distf='dbhat',
           reduction='sum', loc='inpca_results', chunks=1, proj=False):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dr = d1[idx]
    dc = d2[idx]
    if extra_pts is not None:
        qc = extra_pts.loc[:, extra_pts.columns.isin(idx)]
        dc = pd.concat([dc, qc])
        q = th.Tensor(np.stack([extra_pts.iloc[i][key][ss]
                      for i in range(len(extra_pts))])).cpu()
    th.save({'dr': dr, 'dc': dc}, os.path.join(loc, 'didx_%s.p' % fn))
    n, m = len(d1), len(d2)  # number of models

    x = th.Tensor(np.stack([d1.iloc[i][key][ss] for i in range(n)])).cpu()
    y = th.Tensor(np.stack([d2.iloc[i][key][ss] for i in range(m)])).cpu()

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        if 'kl' in distf:
            w = getattr(distance, distf)(x, y, reduction=reduction,
                                         dev=dev, chunks=chunks, probs=probs)
        else:
            x = th.exp(x) if not probs else x
            y = th.exp(y) if not probs else y
            w = getattr(distance, distf)(
                x, y, reduction=reduction, dev=dev, chunks=chunks)
        # w = dist_(x, probs=probs, dev=dev, distf=distf, reduction=reduction)
        print('Saving w')
        th.save(w, os.path.join(loc, 'w_%s.p' % fn))
    else:
        print('Found: ', os.path.join(loc, 'w_%s.p' % fn))
        w = th.load(os.path.join(loc, 'w_%s.p' % fn))

    if proj:
        d_mean = w.mean(0)
        l = np.eye(w.shape[0]) - 1.0/w.shape[0]
        w = -l @ w @ l / 2
        r = proj_(w, n, ne)
        if extra_pts is not None:
            q = lazy_embed(
                q, x, w, d_mean, evals=r['e'], evecs=r['v'], distf=distf, ne=ne, chunks=chunks)
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
                logaa[logaa == -th.inf] = 100
                w_ = (aa * logaa).sum(-1, keepdim=True) - \
                    th.bmm(aa, logaa.transpose(1, 2))
            else:
                w_ = (th.exp(aa) * aa).sum(-1, keepdim=True) - \
                    th.bmm(th.exp(aa), (aa).transpose(1, 2))
            w += getattr(w_, reduction)(0).cpu().numpy()
        elif distf == 'skl':
            if probs:
                logaa = th.log(aa)
                logaa[logaa == -th.inf] = 100
                w1 = (aa * logaa).sum(-1, keepdim=True) - \
                    th.bmm(aa, logaa.transpose(1, 2))
                w2 = (aa * logaa).sum(-1, keepdim=True) - \
                    th.bmm(logaa, aa.transpose(1, 2))
                w_ = 0.5*(w1+w2)
            else:
                w1 = (th.exp(aa) * aa).sum(-1, keepdim=True) - \
                    th.bmm(th.exp(aa), (aa).transpose(1, 2))
                w2 = (th.exp(aa) * aa).sum(-1, keepdim=True) - \
                    th.bmm(aa, th.exp(aa).transpose(1, 2))
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


def process_pair(pair, file_list):
    print(pair)
    s1, s2 = pair
    d1, nan_models = load_d(
        file_list=file_list[s1], avg_err=True, probs=True, return_nan=True)
    th.save(nan_models, f'inpca_results_all/nan_models_{s1}.p')
    if s1 == s2:
        d2 = d1
    else:
        d2, nan_models = load_d(
            file_list=file_list[s2], avg_err=True, probs=True, return_nan=True)
        th.save(nan_models, f'inpca_results_all/nan_models_{s2}.p')

    for key in ['yh', 'yvh']:
        fn = f'{key}_{s1}_{s2}'
        idx = ['seed', 'm', 'opt', 't', 'err', 'favg',
               'verr', 'vfavg', 'bs', 'aug', 'bn', 'lr', 'wd']
        xembed(d1, d2, fn=fn, ss=slice(0, -1, 2), probs=True, key=key,
               idx=idx, force=False, distf='dbhat', reduction='mean', chunks=3600, proj=False)


def main():
    from itertools import combinations
    import torch.multiprocessing as mp
    from functools import partial

    loc = 'results/models/all'
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = defaultdict(list)
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        file_list[(configs["seed"], configs["m"])].append(f)

    load_list_ = list(file_list.keys())
    load_list_ = list(combinations(load_list_, 2)) + \
        [(load_list_[i], load_list_[i]) for i in range(len(load_list_))]

    load_list = []
    for (s1, s2) in load_list_:
        if not os.path.exists(os.path.join('inpca_results',f"w_yh_{s1}_{s2}.p")):
            load_list.append((s1, s2))

    mp.set_start_method('spawn')
    with mp.Pool(processes=16) as pool:
        results = pool.map(
            partial(process_pair, file_list=file_list), load_list, chunksize=1)


def join():
    loc = 'inpca_results'
    key = "yh"
    all_files = glob.glob(os.path.join('results/models/all', '*}.p'))
    file_list = defaultdict(list)
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        file_list[(configs["seed"], configs["m"])].append(f)

    didxs = None
    load_list = list(file_list.keys())

    import h5py 
    f = h5py.File(f'w_{key}_all.h5', 'w') 
    dset = f.create_dataset("w", data=np.zeros([151407, 151407]), maxshape=(None, None))
    start_idx = 0
    for i in range(len(load_list)):
        s1 = load_list[i]
        w_fname = os.path.join(loc, f"w_{key}_{s1}_{s1}.p")
        didxs_fname = os.path.join(loc, f"didx_{key}_{s1}_{s1}.p")
        try:
            w_ = th.load(w_fname)
            didxs = pd.concat([didxs, th.load(didxs_fname)["dc"]])
            bsize = len(w_)
            dset[start_idx:start_idx+bsize, start_idx:start_idx+bsize] = w_
        except FileNotFoundError:
            with open('missed.txt', 'a') as f:
                f.write(f"{w_fname}\n")
        cidx = start_idx
        for j in range(i+1, len(load_list)):
            s2 = load_list[j]
            print(s1, s2)
            fname = os.path.join(loc, f"w_{key}_{s1}_{s2}.p")
            try:
                w_ = th.load(fname)
                l1, l2 = w_.shape
                assert l1 == bsize # debug
                dset[start_idx:start_idx+bsize, cidx:cidx+l2] = w_
                dset[start_idx:start_idx+l2, cidx:cidx+bsize] = w_.T
                cidx += l2
            except FileNotFoundError:
                with open('missed.txt', 'a') as f:
                    f.write(f"{fname}\n")

        start_idx += bsize 
        # print(f"saving iter {i}")
        th.save(didxs, os.path.join(loc, f"didxs_{key}_all.p"))
        # th.save(w, os.path.join(loc, f"w_{key}_all.p"))
    


if __name__ == '__main__':
    # main()
    join()
