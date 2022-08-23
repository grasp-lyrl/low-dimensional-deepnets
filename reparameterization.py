import numpy as np
from tqdm import tqdm

def projection(r, p, q, debug=False, mode='prod'):
    # r, p, q shape (nmodels, nsamples, nclasses)
    # p, q: start and end points of the geodesic
    nm, ns, nc = r.shape
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
        cost1, cost2 = cost1[ti > 0], cost2[ti > 0]
        ti = ti[ti > 0]

        def d(t):
            coss = cost1*np.sin((1-t)*ti) / np.sin(ti) + \
                cost2 * np.sin(t*ti)/np.sin(ti)
            if coss.max() > 1 or coss.min() < 0:
                coss[coss > 1] = 1
                coss[coss < 0] = 0
            return d1 + np.arccos(coss).sum()
        d0 = d(0)
        lams = np.linspace(0, 1, 100)
        lam = np.zeros([nm, 1])
        for (i, dt) in enumerate(map(d, lams)):
            lam[dt < d0] = lams[i]
            d0 = dt
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

def reparam(d, num_ts=50, groups=['m', 'opt', 'seed']):
    new_avg = []
    groups = ['m', 'opt', 'seed']
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
                lam = projection(r, p, q)
                if abs(lam-0.5) < diff:
                    diff = abs(lam-0.5)
                    data['yh'] = (gamma(lam, p, q) ** 2).squeeze()
                    data['err'] = (np.argmax(data['yh'], axis=-1) != y_).mean()
                    data['favg'] = -np.log(data['yh'])[:,
                                                    np.arange(len(y_)), y_].mean()
            # data['yh'] = gamma(lam, p, q)
            print(c, t, k, lam, data['err'], data['favg'])
            new_avg.append(data)
