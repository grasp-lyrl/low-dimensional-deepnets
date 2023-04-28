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

def v0(p, q):
    # p, q shape: nmodels, nsamples, nclasses
    cospq = np.clip((p*q).sum(-1), 0, 1)
    ti = np.arccos(cospq)[:, :, None]
    v0 = -ti*np.cos(ti)/np.sin(ti) * p + ti/np.sin(ti) * q
    return v0

def reparameterize(d, labels, num_ts=50, groups=['m', 'opt', 'seed'], idx='lam', key='yh', idx_lim=(0,1)):
    new_d = []
    configs = d.groupby(groups).indices
    ts = np.linspace(0, 1, (num_ts+1))[1:]
    d[idx] = np.clip(d[idx], *idx_lim)
    d[idx] = d[idx] / (idx_lim[1] - idx_lim[0])
    ind = d.index.min()
    max_ind = d.index.max()
    yhs = []
    errs = []
    favgs = []
    for t in ts:
        k = ind
        while k < max_ind:
            if d.iloc[k][idx] > t:
                break
            k += 1
        if k == max_ind:
            end_lam = 1
        else:
            end_lam = d.iloc[k][idx]
        ind = k
        start = d.iloc[max(0, k-1)]
        end = d.iloc[k]
        if abs(end_lam - start[idx]) < 1e-8:
            yhs.append(start[key])
            errs.append(start['err'])
            favgs.append(start['favg'])
            continue

        lam_interp = (t - start[idx]) / (end_lam - start[idx])
        lam_interp = np.clip(lam_interp, 0, 1)
        r = gamma(lam_interp, np.sqrt(start[key])[None, :], np.sqrt(end[key])[None, :])
        yhs.append((r ** 2).squeeze())
        errs.append((
            np.argmax(yhs[-1], axis=-1) != labels[key]).mean())
        favgs.append(- \
            np.log(yhs[-1])[np.arange(
                len(labels[key])), labels[key]].mean())
    return ts, yhs, errs, favgs


def compute_lambda(file_list, reparam=False, force=False, 
                   didx_loc='inpca_results_all/corners', align_didx='',
                   didx_fn='all', save_loc='results/models/reindexed_new'):

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

    didx_all = None
    cols = ['seed', 'iseed', 'isinit', 'corner', 'aug', 'm', 'opt', 'bs', 'lr', 'wd', 't', 'err', 'verr', 'lam_yh', 'lam_yvh']
    for f in tqdm.tqdm(file_list):
        save_fn = os.path.join(save_loc, os.path.basename(f))
        if os.path.exists(save_fn) and not force:
            continue
        if not os.path.exists(f):
            d = load_d(file_list=[f], avg_err=True, probs=False)
        else:
            try:
                d = load_d(file_list=[f], avg_err=True, probs=False)
            except:
                print(f)
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
                qs_ = np.repeat(qs[key], yhs[key].shape[0], axis=0)
                ps_ = np.repeat(ps[key], yhs[key].shape[0], axis=0)
                d[f'lam_{key}'] = project(yhs[key], ps_, qs_)
            didx_all = pd.concat([didx_all, d.reindex(cols, axis=1)])
            th.save(
                didx_all, os.path.join(didx_loc, f'didx_{didx_fn}.p'))
        else:
            continue


        if reparam:
            d_reparam = pd.DataFrame()
            for key in ['yh', 'yvh']:
                if not probs:
                    d[key] = d.apply(lambda r: np.exp(r[key]), axis=1)
                d['acc'] = 1-d['err']
                ts, yhs, errs, favgs = reparameterize(d, labels, num_ts=100, idx='acc', key=key,
                            groups=['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd'])
                d_reparam[key] = yhs
                d_reparam['t'] = ts
                errkey = 'err_interp' if key == 'yh' else 'verr_interp'
                fkey = 'favg_interp' if key == 'yh' else 'vfavg_interp'
                d_reparam[errkey] = errs
                d_reparam[fkey] = favgs
            r_cols = ['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd']
            d_reparam[r_cols] = d[r_cols].iloc[0]
            th.save(d_reparam, save_fn)
    if align_didx:
        import ipdb; ipdb.set_trace()
        d2 = th.load(os.path.join(didx_loc, align_didx))
        d = didx_all.merge(d2, on=list(didx_all.columns))
        th.save(d, os.path.join(didx_loc, f'didx_{didx_fn}.p'))

if __name__ == '__main__':

    loc = 'results/models/corners'
    fs_all = glob.glob(os.path.join(loc, '*}.p'))
    fs = []

    for f in fs_all:
        configs = json.loads(f[f.find("{"): f.find("}") + 1])
        if not configs['isinit']:
            fs.append(f)

    fs.extend([f'/home/ubuntu/ext_vol/inpca/results/models/loaded/{{"seed":{s},"bseed":-1,"aug":"none","m":"allcnn","bn":true,"drop":0.0,"opt":"sgd","bs":200,"lr":0.1,"wd":0.0,"corner":"normal","interp":false}}.p' for s in range(42, 52)])
    print(len(fs))

    compute_lambda(fs, reparam=False, force=False, 
                   save_loc='results/models/all', 
                   didx_loc='inpca_results_all/corners',
                   didx_fn='noinit_all',
                   align_didx='didx_yh_noinit_with_normal.p'
                )
