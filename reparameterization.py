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
    cost, cost1, cost2 = (p*q).sum(-1, keepdims=True), (p*r).sum(-1, keepdims=True), (q*r).sum(-1, keepdims=True)
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

        def dd(t, n=1):
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
            darccos = - 1. / (np.sqrt(1 - coss**2) + eps)
            dcos = cost1_ / sinti_ * np.cos((1-t)*ti_) * (-ti_) + cost2_ / sinti_ * np.cos(t*ti_) * ti_
            darccos[ii_] = 0
            return (darccos * dcos).sum(1)
            
        lam = []
        for n in range(len(ti)):
            dn = partial(d, n=n)
            l = optimize.minimize_scalar(dn, bounds=(0, 1), method='bounded').x
            lam.append(l)
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

def reparam(d, ps, qs, labels, num_ts=50, groups=['m', 'opt', 'seed']):
    new_d = []
    configs = d.groupby(groups).indices
    ts = np.linspace(0, 1, (num_ts+1))[1:]
    for (c, idx) in configs.items():
        di = d.iloc[idx]
        for t in ts:
            data = {groups[i]: c[i] for i in range(len(c))}
            data['t'] = t
            k1 = di[di['lam'] >= t]['t']
            k2 = di[di['lam'] < t]['t']
            ks = set(k2.index).intersection(
                set(k1.index-1)).intersection(set(di.index[:-1]))
            if len(ks) == 0:
                # continue
                ks = set(di.index[-1:]-1)
            diff = 1
            for k in ks:
                p = np.sqrt(di.loc[k]['yh'])[None, :]
                q = np.sqrt(di.loc[k+1]['yh'])[None, :]
                r = gamma(t, ps, qs)
                lam = float(project(r, p, q)[0])
                if abs(lam-0.5) < diff:
                    diff = abs(lam-0.5)
                    data['yh'] = (gamma(lam, p, q) ** 2).squeeze()
                    data['err'] = (np.argmax(data['yh'], axis=-1) != labels).mean()
                    data['favg'] = -np.log(data['yh'])[np.arange(len(labels)), labels].mean()
            print(c, t, k, lam, data['err'], data['favg'])
            new_d.append(data)
    return new_d


def main():
    loc = 'results/models/all'
    all_files = glob.glob(os.path.join(loc, '*}.p'))
    file_list = []
    for f in all_files:
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if configs['seed'] == 47: 
            file_list.append(f)
    print(len(file_list))

    ds = get_data()['train']
    y_ = np.array(ds.targets, dtype=np.int32)
    y = np.zeros((y_.size, y_.max()+1))
    y[np.arange(y_.size), y_] = 1
    qs = np.sqrt(np.expand_dims(y, axis=0))

    for f in tqdm(file_list):
        save_fn = os.path.join('results/models/loaded', os.path.basename(f))
        if os.path.exists(save_fn):
            d = th.load(save_fn)
            ps = np.sqrt(np.ones([1, 50000, 10]) / 10)
            new_d = reparam(d, ps, qs, y_, num_ts=50,
                            groups=['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd'])
            th.save(new_d, save_fn)
        else:
            d = load_d(file_list=[f], avg_err=True, probs=True)
            if d is not None:
                yhs = np.sqrt(np.stack(d['yh'].values))
                ps = np.sqrt(np.ones_like(yhs) / 10)
                qs_ = np.repeat(qs, yhs.shape[0], axis=0)
                d['lam'] = project(yhs, ps, qs_)
                th.save(d, save_fn)

if __name__ == '__main__':
    main()
