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
    cost = np.clip((p*q).sum(-1, keepdims=True), 0, 1)
    cost1 = np.clip((p*r).sum(-1, keepdims=True), 0, 1)
    cost2 = np.clip((q*r).sum(-1, keepdims=True), 0, 1)

    if mode == 'prod':
        ti = np.arccos(cost)
        mask = ti < eps
        d1 = np.arccos(cost1[mask]).sum()
        sinti = np.sin(ti)

        def d(t, n=1):
            maskn = mask[n, :]
            cost1_ = cost1[n:n+1, ~maskn]
            ti_ = ti[n:n+1, ~maskn]
            cost1_ = cost1[n:n+1, ~maskn]
            cost2_ = cost2[n:n+1, ~maskn]
            sinti_ = sinti[n:n+1, ~maskn]
            coss = cost1_*np.sin((1-t)*ti_) / sinti_ + \
                cost2_ * np.sin(t*ti_)/sinti_
            coss = np.clip(coss, 0, 1)
            t_ = np.arccos(coss)
            return d1 + t_.sum(1)

        lam = []
        for n in range(len(ti)):
            dn = partial(d, n=n)
            l = optimize.minimize_scalar(dn, bounds=(0, 1), method='bounded').x
            lam.append(float(l))
    elif mode == 'mean':
        tan = cost2/(cost1*np.sqrt(1-cost**2)) - cost / np.sqrt(1-cost**2)
        lam = (np.arctan(tan) * (tan > 0)) / np.arccos(cost)
        lam = np.clip(lam, 0, 1)

    return lam


def gamma(t, p, q):
    # p, q shape: nmodels, nsamples, nclasses
    cospq = np.clip((p*q).sum(-1), 0, 1)
    ti = np.arccos(cospq)
    mask = ti < 1e-8
    gamma = np.zeros_like(p)
    gamma[mask, :] = p[mask, :]
    p, q = p[~mask, :], q[~mask, :]
    ti = ti[~mask, None]
    gamma[~mask, :] = np.sin((1-t)*ti) / np.sin(ti) * \
        p + np.sin(t*ti) / np.sin(ti) * q
    return gamma


def reparam(d, labels, num_ts=50, groups=['m', 'opt', 'seed'], key='yh'):
    new_d = []
    configs = d.groupby(groups).indices
    ts = np.linspace(0, 1, (num_ts+1))[1:]
    for (c, idx) in configs.items():
        di = d.iloc[idx]
        ind = {'yh': di.index.min(), 'yvh': di.index.min()}
        max_ind = di.index.max()
        for t in ts:
            data = {groups[i]: c[i] for i in range(len(c))}
            data['t'] = t
            for key in ['yh', 'yvh']:
                k = ind[key]
                while k < max_ind:
                    if di.iloc[k][f'lam_{key}'] > t:
                        break
                    k += 1
                if k == max_ind:
                    end_lam = 1
                else:
                    end_lam = di.iloc[k][f'lam_{key}']
                ind[key] = k
                start = di.iloc[max(0, k-1)]
                end = di.iloc[k]

                if abs(end_lam - start[f'lam_{key}']) < 1e-8:
                    import ipdb; ipdb.set_trace()
                lam_interp = (t - start[f'lam_{key}']) / (end_lam - start[f'lam_{key}'])
                lam_interp = np.clip(lam_interp, 0, 1)
                r = gamma(lam_interp, np.sqrt(start[key])[None, :], np.sqrt(end[key])[None, :])
                data[key] = (r ** 2).squeeze()
                errkey = 'err' if key == 'yh' else 'verr'
                fkey = 'favg' if key == 'yh' else 'vfavg'
                data[errkey] = (
                    np.argmax(data[key], axis=-1) != labels[key]).mean()
                data[fkey] = - \
                    np.log(data[key])[np.arange(
                        len(labels[key])), labels[key]].mean()
            new_d.append(data)
    return pd.DataFrame(new_d)
                ##################### old #########################
                # diff = 1
                # for k in ks:
                #     p = np.sqrt(di.loc[k][key])[None, :]
                #     q = np.sqrt(di.loc[k+1][key])[None, :]
                #     r = gamma(t, ps[key], qs[key])
                #     lam = project(r, p, q)[0]
                #     if abs(lam-0.5) < diff:
                #         diff = abs(lam-0.5)
                #         data[key] = (gamma(lam, p, q) ** 2).squeeze()
                #         errkey = 'err' if key == 'yh' else 'verr'
                #         fkey = 'favg' if key == 'yh' else 'vfavg'
                #         data[errkey] = (
                #             np.argmax(data[key], axis=-1) != labels[key]).mean()
                #         data[fkey] = - \
                #             np.log(data[key])[np.arange(
                #                 len(labels[key])), labels[key]].mean()
                #     print(c, t, k, lam, data[errkey], data[fkey])
                ##################### old #########################


def compute_lambda(reparam=False, force=False, separate=False, didx_fn='all', loc='results/models/all', save_loc='results/models/reindexed_new'):
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = []
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if configs.get('corner') == 'normal':
            file_list.append(f)

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
    if not separate:
        qs = np.concatenate([qs['yh'], qs['yvh']], axis=1)
        ps = np.concatenate([ps['yh'], ps['yvh']], axis=1)
        labels = np.hstack([labels['yh'], labels['yvh']])

    didx_all = None
    cols = ['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd', 't', 'err', 'verr', 'favg',
            'vfavg']
    if separate:
        cols = cols + ['lam_yh', 'lam_yvh']
    else:
        cols = cols + ['lam']
    for f in tqdm.tqdm(file_list):
        load_fn = os.path.join('results/models/loaded', os.path.basename(f))
        save_fn = os.path.join(save_loc, os.path.basename(f))
        if os.path.exists(save_fn) and not force:
            continue
        if not os.path.exists(load_fn):
            d = load_d(file_list=[f], avg_err=True, probs=False)
        else:
            try:
                d = th.load(load_fn)
            except:
                print(load_fn)
                continue
        if d is not None:
            yhs = {}
            for key in ['yh', 'yvh']:
                yhs[key] = np.stack(d[key].values)
                if not np.allclose(yhs[key].sum(-1), 1):
                    yhs[key] = np.exp(yhs[key])
                    probs = False
                else:
                    probs = True
                yhs[key] = np.sqrt(yhs[key])
                if separate:
                    qs_ = np.repeat(qs[key], yhs[key].shape[0], axis=0)
                    ps_ = np.repeat(ps[key], yhs[key].shape[0], axis=0)
                    d[f'lam_{key}'] = project(yhs, ps_, qs_)
            if not separate:
                # yhs = np.stack([yhs['yh'], yhs['yvh']])
                yhs = np.concatenate([yhs['yh'], yhs['yvh']], axis=1)
                qs_ = np.repeat(qs, yhs.shape[0], axis=0)
                ps_ = np.repeat(ps, yhs.shape[0], axis=0)
                d['lam'] = project(yhs, ps_, qs_)
            th.save(d, load_fn)
            didx_all = pd.concat([didx_all, d[cols]])
            th.save(
                didx_all, f'/home/ubuntu/ext_vol/inpca/results/models/loaded/didx_{didx_fn}.p')
        else:
            continue
        if reparam:
            for key in ['yh', 'yvh']:
                if not probs:
                    d[key] = d.apply(lambda r: np.exp(r[key]), axis=1)
            new_d = reparam(d, labels, num_ts=100,
                            groups=['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd'])
            th.save(new_d, save_fn)


if __name__ == '__main__':
    # compute_lambda(reparam=False, force=True)
    compute_lambda(reparam=False, force=False, separate=False, loc='results/models/all', save_loc='results/models/reindexed_all', didx_fn='nonsep')
