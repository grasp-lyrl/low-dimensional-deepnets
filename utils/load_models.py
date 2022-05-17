import os
import json
import glob
import pandas as pd
import torch as th


def get_idx(dd, cond):
    return dd.query(cond).index.tolist()


def load_d(loc, cond={}, avg_err=False, numpy=True, probs=False, drop=0, keys=['yh', 'yvh']):
    r = []
    for f in glob.glob(os.path.join(loc, '*}.p')):
        configs = json.loads(f[f.find('{'):f.find('}')+1])
        if all(configs[k] in v for (k, v) in cond.items()):
            d = th.load(f)
            if isinstance(d, dict):
                d = d['data']
            print(f, len(d))
            ts = list(range(0, 20))+list(range(20, 200, 20))+[200] if len(d) == 201 else range(len(d))
            # for i in range(len(d)):
            for i in range(len(ts)):
                t = {}
                t.update(configs)
                t.update({'t': i / len(ts)})
                t.update(d[ts[i]])
                r.append(t)

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
        d = drop_untrained(d, key='err', th=drop, verbose=False).reset_index()

    print(d.keys(), len(d))
    del r

    return d


def drop_untrained(dd, key='err', th=0.01, verbose=False):
    tmax = dd['t'].max()
    ii = get_idx(dd, f"t == {tmax} & {key} > {th}")
    iis = [j for i in ii for j in range(i-tmax, i+1, 1)]
    if verbose:
        print(len(ii))
        print(dd[['m', 'opt', 'bn', 'seed']].iloc[ii])
    return dd.drop(iis)
