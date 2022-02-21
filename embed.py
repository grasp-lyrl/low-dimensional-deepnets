from utils import setup, get_data
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


def get_idx(dd, cond):
    return dd.query(cond).index.tolist()


def embed(dd, fn='', ss=slice(0,-1,1), ne=3, key='yh', force=False, idx=None, dev='cuda'):
    idx = idx or ['seed', 'widen', 'numc', 't', 'err', 'verr', 'favg', 'vfavg']
    dc = dd[idx]
    th.save(dc, os.path.join(loc, 'didx_%s.p' % fn))
    # x = np.array([dd.iloc[i][key][::ss].float().numpy()
    #              for i in range(len(dd))])
    # n = x.shape[0]
    n = len(dd)
    x = th.stack([dd.iloc[i][key][ss].float() for i in range(n)]).to(dev)

    if (not os.path.isfile(os.path.join(loc, 'w_%s.p' % fn))) or force:
        # a = np.sqrt(np.exp(x))
        # a = np.moveaxis(a, 0, 1)
        a = th.sqrt(th.exp(x))
        a = th.moveaxis(a, 0, 1)
        w = np.zeros((n, n))
        nc = 100 if n < 4000 else 600
        print('chunks: ', nc)
        # for aa in tqdm.tqdm(np.split(a, nc)):
        for aa in tqdm.tqdm(th.chunk(a, nc)):
            # w += np.log(np.einsum('kil,kjl->kij', aa,
            #             aa, optimize=True)).sum(0)
            w_ = th.log(th.bmm(aa, aa.transpose(1, 2)))
            w_[w_ == -th.inf] = -100
            w_[w_ > 0] = 0
            w += w_.sum(0).cpu().numpy()
        w = -w
        print('Saving w')
        th.save(w, os.path.join(loc, 'w_%s.p' % fn))

        del a
        l = np.eye(w.shape[0]) - 1.0/w.shape[0]
        w = -l @ w @ l / 2
    else:
        print('Found: ', os.path.join(loc, 'w_%s.p' % fn))

    w = th.load(os.path.join(loc, 'w_%s.p' % fn))
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
    r = dict(xp=xp, w=w, e=e, v=v)
    th.save(r, os.path.join(loc, 'r_%s.p' % fn))
    return

dev = 'cuda'

loc = 'results/models/new'

r = []
for f in glob.glob(os.path.join(loc, '*}.p')):

    k = json.loads(f[f.find('{'):f.find('}')+1])
    if k['bs'] == 200 and k['aug']:
        d = th.load(f)
        print(f, len(d))
        for i in range(len(d)):
            t = {}
            t.update(k)
            t.update({'t': i})
            t.update(d[i])
            r.append(t)

d = pd.DataFrame(r)
d['err'] = d.apply(lambda r: r.e.mean().item(), axis=1)
d['verr'] = d.apply(lambda r: r.ev.mean().item(), axis=1)
d['favg'] = d.apply(lambda r: r.f.mean().item(), axis=1)
d['vfavg'] = d.apply(lambda r: r.fv.mean().item(), axis=1)

print(d.keys(), len(d))
del r

setup(2)
ds = get_data(dev='cuda', aug=True)

import ipdb; ipdb.set_trace()
# for t in range(20):
#     for bn in [True, False]:
#         print(t, bn)
#         ii = get_idx(d, f"bn == {bn} & t == {t}")
#         yh = th.stack(list(d.iloc[ii]['yh'].values))
#         yhavg = th.mean(yh, axis=0)
#         yvh = th.stack(list(d.iloc[ii]['yvh'].values))
#         yvhavg = th.mean(yvh, axis=0)
#         avg_m = pd.Series({'m': 'avg', 'bn': bn, 't': t, 
#                         'yh': yhavg, 'e': th.argmax(yhavg, axis=1) == ds['y'].cpu(),
#                         'yvh': yvhavg, 'ev': th.argmax(yvhavg, axis=1) == ds['yv'].cpu()})
#         d = d.append(avg_m, ignore_index=True) 

loc = 'inpca_results'
idx = ['seed', 'm', 'opt', 't', 'err', 'verr', 'favg', 'vfavg', 'bn', 'bs', 'aug']
for key in ['yh']:
    fn = f'{key}_new_aug'
    # idxs = th.load(os.path.join(loc, f'{key}_idx.p'))
    embed(d, fn=fn, ss=slice(0,-1,5), key=key, idx=idx, force=True)
