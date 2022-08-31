import numpy as np
from scipy import optimize
from tqdm import tqdm
import glob
from functools import partial
from utils import *


def project(r, p, q, debug=False, mode='prod'):
    # r, p, q shape (nmodels, nsamples, nclasses)
    # p, q: start and end points of the geodesic
    nm, ns, nc = r.shape
    eps = 1e-8
    if debug:
        assert np.allclose([(r**2).sum(-1), (p**2).sum(-1), (q**2).sum(-1)], 1)
    cost, cost1, cost2 = (p*q).sum(-1, keepdims=True), (p *
                                                        r).sum(-1, keepdims=True), (q*r).sum(-1, keepdims=True)
    if mode == 'prod':
        if cost.max() > 1 or cost.min() < 0:
            cost[cost > 1] = 1
            cost[cost < 0] = 0
        ti = np.arccos(cost)
        d1 = np.arccos(cost1[ti == 0]).sum()
        sinti = np.sin(ti)
        ii = sinti == 0
        sinti[ii] += eps

        def d(t, n=1):
            cost1_ = cost1[n:n+1, :]
            ti_ = ti[n:n+1, :]
            ii_ = ii[n:n+1, :]
            cost1_ = cost1[n:n+1, :]
            cost2_ = cost2[n:n+1, :]
            sinti_ = sinti[n:n+1, :]
            coss = cost1_*np.sin((1-t)*ti_) / sinti_ + \
                cost2_ * np.sin(t*ti_)/sinti_
            if coss.max() > 1 or coss.min() < 0:
                coss[coss > 1] = 1
                coss[coss < 0] = 0
            t_ = np.arccos(coss)
            t_[ii_] = 0
            return d1 + t_.sum(1)

        lam = []
        for n in range(len(ti)):
            dn = partial(d, n=n)
            l = optimize.minimize_scalar(dn, bounds=(0, 1), method='bounded').x
            lam.append(float(l))
    elif mode == 'mean':
        tan = cost2/(cost1*np.sqrt(1-cost**2)) - cost / np.sqrt(1-cost**2)
        lam = (np.arctan(tan) * (tan > 0)) / np.arccos(cost)
        lam[lam > 1] = 1
    return lam


def gamma(t, p, q):
    # p, q shape: nmodels, nsamples, nclasses
    cospq = (p*q).sum(-1)
    if cospq.max() > 1 or cospq.min() < 0:
        cospq[cospq > 1] = 1
        cospq[cospq < 0] = 0
    ti = np.arccos(cospq)
    mask = ti == 0
    gamma = np.zeros_like(p)
    gamma[mask, :] = p[mask, :]
    p, q = p[~mask, :], q[~mask, :]
    ti = ti[~mask, None]
    gamma[~mask, :] = np.sin((1-t)*ti) / np.sin(ti) * \
        p + np.sin(t*ti) / np.sin(ti) * q
    return gamma


def reparam(d, ps, qs, labels, num_ts=50, groups=['m', 'opt', 'seed'], key='yh'):
    new_d = []
    configs = d.groupby(groups).indices
    ts = np.linspace(0, 1, (num_ts+1))[1:]
    for (c, idx) in configs.items():
        di = d.iloc[idx]
        for t in ts:
            data = {groups[i]: c[i] for i in range(len(c))}
            data['t'] = t
            for key in ['yh', 'yvh']:
                k1 = di[di[f'lam_{key}'] >= t]['t']
                k2 = di[di[f'lam_{key}'] < t]['t']
                ks = set(k2.index).intersection(
                    set(k1.index-1)).intersection(set(di.index[:-1]))
                if len(ks) == 0:
                    ks = set(di.index[-1:]-1)
                diff = 1
                for k in ks:
                    p = np.sqrt(di.loc[k][key])[None, :]
                    q = np.sqrt(di.loc[k+1][key])[None, :]
                    r = gamma(t, ps[key], qs[key])
                    lam = project(r, p, q)[0]
                    if abs(lam-0.5) < diff:
                        diff = abs(lam-0.5)
                        data[key] = (gamma(lam, p, q) ** 2).squeeze()
                        errkey = 'err' if key == 'yh' else 'verr'
                        fkey = 'favg' if key == 'yh' else 'vfavg'
                        data[errkey] = (
                            np.argmax(data[key], axis=-1) != labels[key]).mean()
                        data[fkey] = - \
                            np.log(data[key])[np.arange(
                                len(labels[key])), labels[key]].mean()
                    print(c, t, k, lam, data[errkey], data[fkey])
            new_d.append(data)
    return pd.DataFrame(new_d)


def main():
    loc = 'results/models/all'
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = all_files
    # file_list = []
    # for f in all_files:
    #     load_fn = os.path.join('results/models/loaded', os.path.basename(f))
    #     if not os.path.exists(load_fn):
    #         file_list.append(f)

    data = get_data()
    labels = {}
    qs = {}
    ps = {}
    for key in ['train', 'val']:
        k = 'yh' if key == 'train' else 'yvh'
        y_ = np.array(data[key].targets, dtype=np.int32)
        y = np.zeros((y_.size, y_.max()+1))
        y[np.arange(y_.size), y_] = 1
        qs[k] = np.sqrt(np.expand_dims(y, axis=0))
        ps[k] = np.sqrt(np.ones_like(qs[k]) / 10)
        labels[k] = y_

    for f in tqdm.tqdm(file_list):
        load_fn = os.path.join('results/models/loaded', os.path.basename(f))
        save_fn = os.path.join('results/models/reindexed', os.path.basename(f))
        if not os.path.exists(load_fn):
            d = load_d(file_list=[f], avg_err=True, probs=False)
        else:
            d = th.load(load_fn)
        if d is not None and 'lam' not in d.columns:
            for key in ['yh', 'yvh']:
                yhs = np.sqrt(np.exp(np.stack(d[key].values)))
                qs_ = np.repeat(qs[key], yhs.shape[0], axis=0)
                d[f'lam_{key}'] = project(yhs, ps[key], qs_)
                th.save(d, load_fn)
        elif d is None:
            continue
        for key in ['yh', 'yvh']:
            d[key] = d.apply(lambda r: np.exp(r[key]), axis=1)
        new_d = reparam(d, ps, qs, labels, num_ts=100,
                        groups=['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd'])
        th.save(new_d, save_fn)


def avg_by_reindex():
    groups = ['aug', 'm', 'opt', 'bs', 'lr', 'wd']
    didx = th.load(
        '/home/ubuntu/results/inpca/inpca_results_all/didxs_yh_all.p')
    indices = didx.groupby(groups).indices
    loc = 'results/models/reindexed/'
    for k in tqdm.tqdm(indices.keys()):
        if k[1] == 'random' or k[1] == 'true':
            continue
        d = None
        for seed in range(42, 52):
            fdict = dict(seed=seed, bseed=-1, aug=k[0],
                     m=k[1], bn=True, drop=0.0, opt=k[2],
                     bs=k[3], lr=k[4], wd=k[5])
            fn = json.dumps(fdict).replace(' ', '')
            if not os.path.exists(os.path.join(loc, f"{fn}.p")):
                continue
            else:
                d_ = th.load(os.path.join(loc, f"{fn}.p"))
                d = pd.concat([d, d_])
        fdict['seed'] = -1
        fn = os.path.join(loc, f"{json.dumps(fdict).replace(' ', '')}.p")

        d_avg = avg_model(d, probs=True, groupby=groups, update_d=False)
        
        print('saving ', fn)
        th.save(d_avg['avg'], fn)


if __name__ == '__main__':
    # main()
    avg_by_reindex()
