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

from embed import *

def get_idxs(s, file_list, idx=None):
    print(s)
    idx = idx or ['seed', 'm', 'opt', 't', 'err', 'favg',
                  'verr', 'vfavg', 'bs', 'aug', 'bn', 'lr', 'wd']
    d = load_d(file_list=file_list[s],
               avg_err=True, probs=False, return_nan=False)
    didx = d[idx]
    th.save(didx, os.path.join('inpca_results', f'didx_{s}.p'))
    print(f'saved {s}')

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
            file_list=file_list[s1], avg_err=False, probs=True, numpy=False, return_nan=False, loaded=True)
    if s1 == s2:
        d2 = d1
    else:
        # d2, nan_models = load_d(
        # file_list=file_list[s2], avg_err=False, probs=True, return_nan=True, loaded=True)
        # th.save(nan_models, f'inpca_results_avg/nan_models_{s2}.p')
        d2 = load_d(
            file_list=file_list[s1], avg_err=False, probs=True, numpy=False, return_nan=False, loaded=True)

    for key in ['yh', 'yvh']:
        fn = f'{key}_{s1}_{s2}'
        idx = ['seed', 'm', 'opt', 't', 'err', 'favg',
               'verr', 'vfavg', 'bs', 'aug', 'lr', 'wd']
        xembed(d1, d2, fn=fn, ss=slice(0, -1, 2), probs=True, key=key, loc=loc,
               idx=idx, force=False, distf='dbhat', reduction='mean', chunks=3600, proj=False)


def compute_distance(file_list, load_list, loc='results/models/reindexed', groupby=['seed', 'm'], save_loc='inpca_results_avg'):
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = defaultdict(list)
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        file_list[(configs[key] for key in groupby)].append(f)

    load_list_ = product(file_list.keys(), file_list.keys())
    load_list = []
    for pair in load_list_:
        s1, s2 = pair
        if not os.path.exists(os.path.join(save_loc, f'didx_yh_{s1}_{s2}.p')):
            load_list.append(pair)
    print(len(load_list))

    mp.set_start_method('spawn')
    with mp.Pool(processes=2) as pool:
        pool.map(
            partial(process_pair, file_list=file_list, loc=save_loc), load_list, chunksize=1)


def join():
    loc = 'inpca_results'
    key = "yh"
    seeds = range(42, 52)
    m = CHOICES['m']
    r_list = ['end_points']
    c_list = list(product(seeds, m))  # existing
    save_loc = 'inpca_results_all'

    didxs_f = os.path.join(save_loc, f"didxs_{key}_all.p")
    if os.path.exists(didxs_f):
        didxs = th.load(didxs_f)
    else:
        didxs = None
    indices = didxs.groupby(['seed', 'm']).indices
    for r in r_list:
        didx_ = th.load(os.path.join(loc, f"didx_{r}.p"))
        didx_idx = didx_.groupby(['seed', 'm']).indices
        for k in didx_idx.keys():
            if indices.get(k) is None:
                didxs = pd.concat([didxs, didx_.iloc[didx_idx[k]]])
        print(len(didxs))
    th.save(didxs, didxs_f)

    fname = os.path.join(save_loc, f'w_{key}_all.h5')
    if os.path.exists(fname):
        f = h5py.File(fname, 'r+')
        dset = f['w']
        dset.resize((len(r_list) + len(dset), len(r_list) + len(dset)))
    else:
        f = h5py.File(fname, 'w')
        dset = f.create_dataset("w", shape=(len(didxs), len(
            didxs)), maxshape=(None, None), chunks=True)

    for r in r_list:
        w_fname = os.path.join(loc, f"w_{key}_{r}_{r}.p")
        w_ = th.load(w_fname)
        if r == 'end_points':
            ridxs = list(indices[(0, 'true')]) + list(indices[(0, 'random')])
        else:
            ridxs = indices[r]

        def is_cts(l): return all((np.array(l[1:]) - np.array(l[:-1])) == 1)
        assert is_cts(ridxs)  # debug
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


def project(seed=42, fn='yh_all', err_threshold=0.1, extra_points=None):
    loc = "inpca_results_all"
    ne = 3

    folder = os.path.join(loc, str(seed))
    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.isfile(os.path.join(folder, f'didx_{fn}.p')):
        idx, didx = th.load(os.path.join(folder, f'didx_{fn}.p'))
    else:
        # filter out un-trained model
        idx = []
        didx = th.load(os.path.join(loc, f"didxs_{fn}.p"))
        for (c, indices) in didx.groupby(['seed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd']).indices.items():
            if (didx.iloc[indices[-1]]['err'] < err_threshold and c[0] == seed) or c[0] == 0:
                idx.extend(indices)
        idx = sorted(idx)
        didx = didx.iloc[idx]
        th.save((idx, didx), os.path.join(folder, f'didx_{fn}.p'))

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
        import ipdb
        ipdb.set_trace()
        didx_ = th.load(os.path.join(loc, f"didxs_{fn}.p"))
        ridx = get_idx(didx_, extra_points)
        dp = f['w'][:, idx][ridx, :]
        q = lazy_embed(dp=dp, d_mean=d_mean, evals=r['e'], evecs=r['v'], ne=ne)
        r['xp'] = np.vstack([r['xp'], q])
        didx = pd.concat([didx, didx_.iloc[ridx]])
        th.save((idx + ridx, didx), os.path.join(folder, f'didx_{fn}.p'))

    th.save(r, os.path.join(folder, f'r_{fn}_all.p'))


def compute_path_distance_(loc='results/models/reindexed', save_loc='inpca_results_avg',
                          sym='mean', normalization='length', key='yh'):

    all_files = glob.glob(os.path.join(loc, '*}.p'))
    didx = None
    idx = ['seed', 'm', 'opt', 't', 'err', 'favg',
           'verr', 'vfavg', 'bs', 'aug', 'lr', 'wd']
    dists = np.zeros([len(all_files), len(all_files)])
    for i in tqdm.tqdm(range(len(all_files))):
        d1 = load_d(file_list=all_files[i:i+1], avg_err=False,
                    probs=False, numpy=False, return_nan=False, loaded=True)
        didx = pd.concat([didx, d1[idx]])
        th.save(didx, os.path.join(
            save_loc, f'didx_{sym}_{normalization}_{key}.p'))
        x1 = np.stack(d1[key])[None, :]
        for j in range(i, len(all_files)):
            d2 = load_d(file_list=all_files[j:j+1], avg_err=False,
                        probs=False, numpy=False, return_nan=False, loaded=True)
            x2 = np.stack(d2[key])[None, :]
            if sym == 'none':
                distxy, distyx = dt2t_batch(th.Tensor(x1), th.Tensor(
                    x2), reduction='mean', sym=sym, normalization=normalization)
                dists[i, j] = distxy.flatten().cpu().numpy()
                dists[j, i] = distyx.flatten().cpu().numpy()
            elif sym == 'pointwise':
                dists[i, j] = dbhat(th.Tensor(x1.squeeze()), th.Tensor(x2.squeeze()), cross_terms=False).mean()
            else:
                dists[i, j] = dt2t_batch(th.Tensor(x1), th.Tensor(
                    x2), reduction='mean', sym=sym, normalization=normalization).flatten().cpu().numpy()
        th.save(dists, os.path.join(
            save_loc, f'dists_{sym}_{normalization}_{key}.p'))

def compute_path_distance(loc='results/models/reindexed', save_loc='inpca_results_avg', key='yh'):
    all_files = np.array(glob.glob(os.path.join(loc, '*}.p')))
    didx = None
    idx = ['seed', 'm', 'opt', 't', 'err', 'favg',
           'verr', 'vfavg', 'bs', 'aug', 'lr', 'wd']
    # dists = np.zeros([len(all_files), len(all_files)])
    dists = th.load(
        '/home/ubuntu/results/inpca/inpca_results_avg/dists_pointwise_yh.p')
    chunks = np.array_split(np.arange(len(all_files)), 50)
    with tqdm.tqdm(total=len(chunks)**2//2) as pbar:
        for i in tqdm.tqdm(range(46, len(chunks))):
            i1 = chunks[i]
            d1 = load_d(file_list=all_files[i1], avg_err=False,
                        probs=False, numpy=False, return_nan=False, loaded=True)
            t1 = d1.groupby(['t']).indices
            # didx = pd.concat([didx, d1.iloc[t1[1.0]][idx]])
            for j in tqdm.tqdm(range(i, len(chunks))):
                i2 = chunks[j]
                d2 = load_d(file_list=all_files[i2], avg_err=False,
                        probs=False, numpy=False, return_nan=False, loaded=True)
                t2 = d2.groupby(['t']).indices
                w = np.zeros([len(i1), len(i2)])
                for (t, ii) in t1.items():
                    x1 = np.stack(d1.iloc[ii][key])
                    x2 = np.stack(d2.iloc[t2[t]][key])
                    w += dbhat(th.Tensor(x1), th.Tensor(x2), reduction='mean', chunks=50).numpy()
                assert np.allclose(dists[i1[0]:i1[-1]+1, i2[0]:i2[-1]+1], 0)
                dists[i1[0]:i1[-1]+1, i2[0]:i2[-1]+1] = w
                pbar.update(1)
                # th.save(didx, os.path.join(
                #     save_loc, f'didx_pointwise_{key}.p'))
                th.save(dists, os.path.join(
                    save_loc, f'dists_pointwise_{key}.p'))


if __name__ == '__main__':
    # compute_distance()
    # join()
    # for seed in [42, 45, 49, 51]:
    #     project(seed, 'yh_all')
    compute_path_distance()
