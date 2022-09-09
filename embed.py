from genericpath import isfile
import numpy as np
import scipy.linalg as sp
import torch as th

from itertools import combinations
import torch.multiprocessing as mp
from functools import partial
from itertools import product
import os
import h5py
import json
import glob
import tqdm
import pandas as pd

from utils import *
from utils import distance


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
        # w = th.load(os.path.join(loc, 'w_%s.p' % fn))

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


# Calculate the embedding of a distibution q in the intensive embedding of models ps with divergence=distf, supply d_list the precalculated matrix of distances pf p_list.
def lazy_embed(q=None, ps=None, w=None, d_mean=None, dp=None, evals=None, evecs=None, distf='dbhat', ne=3, chunks=1):
    # w: centered pairwise distance, d_mean: mean before centering
    if dp is None:
        dp = getattr(distance, distf)(q, ps, chunks=chunks)
    d_mean_mean = np.mean(d_mean)
    if (evals is None) or (evecs is None):
        N, _, _ = ps.shape
        _, _, evals, evecs = proj_(w, N, ne).values()
    dp_mean = dp - np.mean(dp, 1, keepdims=True) - d_mean + d_mean_mean
    dp_mean = -.5*dp_mean
    sqrtsigma = np.sqrt(np.abs(evals))
    return ((1/sqrtsigma)*np.matmul(dp_mean, evecs))


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


def process_pair(pair, file_list, loc='inpca_results'):
    print(pair)
    s1, s2 = pair
    if s1 == 'end_points' or s2 == 'end_points':
        d1 = th.load(file_list[s1])
    else:
        # d1, nan_models = load_d(
            # file_list=file_list[s1], avg_err=True, probs=True, return_nan=True, loaded=loaded)
        # th.save(nan_models, f'inpca_results_avg/nan_models_{s1}.p')
        d1 = load_d(
            file_list=file_list[s1], avg_err=True, probs=True, return_nan=False, loaded=False)
    if s1 == s2:
        d2 = d1
    else:
        # d2, nan_models = load_d(
            # file_list=file_list[s2], avg_err=False, probs=True, return_nan=True, loaded=True)
        # th.save(nan_models, f'inpca_results_avg/nan_models_{s2}.p')
        d2 = load_d(
            file_list=file_list[s2], avg_err=True, probs=True, return_nan=False, loaded=False)

    for key in ['yh', 'yvh']:
        fn = f'{key}_{s1}_{s2}'
        idx = ['seed', 'm', 'opt', 't', 'err', 'favg',
               'verr', 'vfavg', 'bs', 'aug', 'lr', 'wd']
        xembed(d1, d2, fn=fn, ss=slice(0, -1, 2), probs=True, key=key, loc=loc,
               idx=idx, force=False, distf='dbhat', reduction='mean', chunks=3600, proj=False)


def compute_distance(loc='results/models/all', groupby=['seed', 'm', 'corner'], save_loc='inpca_results'):
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = defaultdict(list)
    corner_list = defaultdict(list)
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if configs['corner'] == 'normal':
            file_list[tuple(configs[key] for key in groupby)].append(f)
        else:
            corner_list[tuple(configs[key] for key in groupby)].append(f)

    # load_list = list(product(corner_list.keys(), file_list.keys()))
    load_list = list(product(corner_list.keys(), corner_list.keys()))
    print(len(load_list))
    file_list.update(corner_list)
    # load_list = []
    # for pair in load_list_:
    #     s1, s2 = pair
    #     if not os.path.exists(os.path.join(save_loc, f'didx_yh_{s1}_{s2}.p')):
    #         load_list.append(pair)
    # print(len(load_list))

    mp.set_start_method('spawn')
    with mp.Pool(processes=2) as pool:
        pool.map(
            partial(process_pair, file_list=file_list, loc=save_loc), load_list, chunksize=1)

def join():
    loc = 'inpca_results'
    key = "yh"
    choices = [('allcnn', 'subsample-200'), ('fc', 'subsample-2000'), ('wr-10-4-8', 'uniform')]
    r_list = []
    for s in range(42, 52):
        r_list.extend([(s, *c) for c in choices])
    c_list = list(product(range(42, 52), CHOICES['m'], ["normal"])) # existing 
    save_loc = 'inpca_results_all'

    didxs_f = os.path.join(save_loc, f"didxs_{key}_all.p")
    if os.path.exists(didxs_f):
        didxs = th.load(didxs_f)
    else:
        didxs = None
    indices = didxs.groupby(['seed', 'm', 'corner']).indices
    for r in r_list:
        didx_ = th.load(os.path.join(loc, f"didx_{r}.p"))
        didx_idx = didx_.groupby(['seed', 'm', 'corner']).indices
        for k in didx_idx.keys():
            if indices.get(k) is None:
                didxs = pd.concat([didxs, didx_.iloc[didx_idx[k]]])
        print(len(didxs))
    th.save(didxs, didxs_f)

    fname = os.path.join(save_loc, f'w_{key}_all.h5')
    if os.path.exists(fname):
        f = h5py.File(fname, 'r+')
        dset = f['w']
    else:
        f = h5py.File(fname, 'w')
        dset = f.create_dataset("w", shape=(len(didxs), len(
            didxs)), maxshape=(None, None), chunks=True)

    indices = didxs.groupby(['seed', 'm', 'corner']).indices
    try:
        for r in r_list:
            w_fname = os.path.join(loc, f"w_{key}_{r}_{r}.p")
            w_ = th.load(w_fname)
            if r == 'end_points':
                ridxs = list(indices[(0, 'true')]) + list(indices[(0, 'random')])
            else:
                ridxs = indices[r]
            dset.resize((len(ridxs)+len(dset), len(ridxs)+len(dset)))
            is_cts = lambda l : all((np.array(l[1:]) - np.array(l[:-1])) == 1)
            assert is_cts(ridxs) # debug
            rstart, rend = ridxs[0], ridxs[-1]+1
            dset[rstart:rend, rstart:rend] = w_

            for c in c_list:
                print(r, c)
                fname = os.path.join(loc, f"w_{key}_{r}_{c}.p")
                if os.path.exists(fname):
                    w_ = th.load(fname)
                else:
                    continue
                if c == 'end_points':
                    cidxs = indices[(0, 'true')] + indices[(0, 'random')]
                else:
                    cidxs = indices[c]
                assert is_cts(cidxs)    
                cstart, cend = cidxs[0], cidxs[-1]+1
                dset[rstart:rend, cstart:cend] = w_
                dset[cstart:cend, rstart:rend] = w_.T
        f.close()
    except:
        import ipdb; ipdb.set_trace()


def project(cond={"seed":[45]}, save_loc='45', fn='yh_all', err_threshold=0.1, extra_points=None, force=False):
    loc = "inpca_results_all"
    ne = 3

    groups = ['seed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd', 'corner']
    folder = os.path.join(loc, save_loc)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if os.path.isfile(os.path.join(folder, f'didx_{fn}.p')) and not force:
        idx, didx = th.load(os.path.join(folder, f'didx_{fn}.p'))
    else:
        # filter out un-trained model
        idx = []
        didx = th.load(os.path.join(loc, f"didxs_{fn}.p"))
        for (c, indices) in didx.groupby(groups).indices.items():
            well_trained = didx.iloc[indices[-1]]['err'] < err_threshold 
            meet_conds = True
            for (i, g) in enumerate(groups):
                meet_conds = meet_conds and ((c[i] in cond[g]) if g in cond.keys() else True)
            if (well_trained and meet_conds) or c[0] == 0:
                idx.extend(indices)
        idx = sorted(idx)
        didx = didx.iloc[idx]
        th.save((idx, didx), os.path.join(folder, f'didx_{fn}.p'))

    print(len(idx))
    print(didx['corner'].unique())
    f = h5py.File(os.path.join(loc, f'w_{fn}.h5'), 'r')
    w = f['w'][:, idx][idx, :]
    n = w.shape[0]
    d_mean = w.mean(0)

    if os.path.isfile(os.path.join(folder, f'r_{fn}.p')):
        r = th.load(os.path.join(folder, f'r_{fn}.p'))
    else:
        l = np.eye(w.shape[0]) - 1.0/w.shape[0]
        w = -l @ w @ l / 2
        r = proj_(w, n, ne)
        th.save(r, os.path.join(folder, f'r_{fn}.p'))

    if extra_points is not None:
        didx_ = th.load(os.path.join(loc, f"didxs_{fn}.p"))
        ridx = get_idx(didx_, extra_points)
        dp = f['w'][:, idx][ridx, :]
        q = lazy_embed(dp=dp, d_mean=d_mean, evals=r['e'], evecs=r['v'], ne=ne)
        r['xp'] = np.vstack([r['xp'], q])
        didx = pd.concat([didx, didx_.iloc[ridx]])
        th.save((idx + ridx, didx), os.path.join(folder, f'didx_{fn}_all.p'))

    th.save(r, os.path.join(folder, f'r_{fn}.p'))

if __name__ == '__main__':
    # cond = {"seed":[43, 46], "aug": ["simple"]}
    cond = {"m":['allcnn', 'fc', 'wr-10-4-8', 'true', 'random'], 
    "corner":["uniform", "subsample-200", "subsample-2000"]}
    project(cond=cond, save_loc='other_corners', fn='yh_all', err_threshold=0.1, extra_points=None)
    # join()
    # compute_distance()
    # for seed in [42, 45, 49, 51]:
    #     project(seed, 'yh_all')
    # project(47, 'yh_all', err_threshold=0.1, extra_points="seed == 42")
