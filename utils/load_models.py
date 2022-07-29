import os
import json
import glob
import pandas as pd
import torch as th
import numpy as np
import math
from collections import defaultdict


ts = defaultdict(list)
for bs in [200, 500, 1000]:
    for epochs in [200, 300]:
        key = (bs, epochs)
        t = 0
        ts[key].append(t)
        n_batches = 50000 // bs
        for epoch in range(epochs):
            for i in range(n_batches):
                t += 1
                if epoch < 5 and i % (n_batches // 4) == 0:
                    ts[(bs, epochs)].append(t)

        if 5 <= epoch <= 25:
            ts[key].append(t)
        elif 25 < epoch <= 65 and epoch % 4 == 0:
            ts[key].append(t)
        elif epoch > 65 and epoch % 15 == 0 or (epoch == epochs-1):
            ts[key].append(t)


def get_idx(dd, cond):
    return dd.query(cond).index.tolist()


def load_d(loc, cond={}, avg_err=False, numpy=True, probs=False, drop=0, keys=['yh', 'yvh'], verbose=False, nmodels=-1, return_nan=False):
    r = []
    nan_models = []
    count = 0
    nmodels = math.inf if nmodels < 0 else nmodels
    for f in glob.glob(os.path.join(loc, '*}.p')):
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if all(configs.get(k, "na") in v for (k, v) in cond.items()):
            d_ = th.load(f)
            d = d_['data']
            if any(np.isnan(d[-1]['f'])):
                nan_models.append(configs)
                continue
            if verbose:
                print(f, len(d))
            for i in range(len(d)):
                t = {}
                t.update(configs)
                t.update({'epochs': getattr(d_['configs'], 'epochs')})
                if ts is not None:
                    t_ = ts[(t['bs'], t['epochs'])]
                else:
                    t_ = i
                t.update({'t': t_})
                t.update(d[i])
                r.append(t)
            count += 1
            if count > nmodels:
                break
    print(len(r))

    d = pd.DataFrame(r)
    if avg_err:
        d['err'] = d.apply(lambda r: r.e.mean().item(), axis=1)
        d['verr'] = d.apply(lambda r: r.ev.mean().item(), axis=1)
        d['favg'] = d.apply(lambda r: r.f.mean().item(), axis=1)
        d['vfavg'] = d.apply(lambda r: r.fv.mean().item(), axis=1)

    for k in keys:
        if probs:
            d[k] = d.apply(lambda r: th.exp(r[k]), axis=1)
        if numpy:
            d[k] = d.apply(lambda r: r[k].numpy(), axis=1)

    if drop:
        d = drop_untrained(d, key='err', th=drop, verbose=True).reset_index()

    print(d.keys(), len(d))
    del r

    if return_nan:
        return d, pd.DataFrame(nan_models)
    else:
        return d


def drop_untrained(dd, key='err', end_t=None, th=0.01, verbose=False):
    tmax = end_t or [dd['t'].max()]
    idxs = {t: get_idx(dd, f"t == {t} & {key} > {th}") for t in tmax}
    iis = []
    for (t, ii) in idxs.items():
        for i in ii:
            iis += list(range(i-t, i+1, 1))
    if verbose:
        print(len(ii))
        print(dd[['m', 'opt', 'bn', 'seed']].iloc[ii])
    return dd.drop(iis)
