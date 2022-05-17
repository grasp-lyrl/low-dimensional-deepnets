import torch as th
import numpy as np

import pandas as pd
from scipy.interpolate import interpn
from utils import get_data

def projection(r, p, q, debug=False):
    # r, p, q shape (npoints, ndim)
    if debug:
        assert np.allclose((r**2).sum(1), (p**2).sum(1), (q**2).sum(1), 1)
    cost, cost1, cost2 = (p*q).sum(1), (p*r).sum(1), (q*r).sum(1)
    tan = cost2/(cost1*np.sqrt(1-cost**2)) - cost / np.sqrt(1-cost**2)
    lam = (np.arctan(tan) * (tan > 0)) / np.arccos(cost)
    lam[lam > 1] = 1
    return lam

def proj_geodesic(d, y):
    pass

def avg_model(d, groupby=['m', 't'], probs=False, avg=None, bootstrap=False, get_err=True, update_d=False, keys=['yh', 'yvh'],
              compute_distance=False, dev='cuda', distf=lambda x, y: th.cdist(x, y).mean(0)):
    d_return = {}

    n_data = {k: d[k].iloc[0].shape[0] for k in keys}
    if avg is None:
        for k in keys:
            if isinstance(d.iloc[0][k], th.Tensor):
                d[k] = d.apply(lambda r: r[k].flatten().numpy(), axis=1)
            else:
                d[k] = d.apply(lambda r: r[k].flatten(), axis=1)
            if not probs:
                d[k] = d.apply(lambda r: np.exp(r[k]), axis=1)
                   
        if bootstrap:
            idxs = d[d['t'] == 0].groupby(groupby).indices
            m, n = len(idxs), len(next(iter(idxs.values())))
            T = len(d['t'].unique())
            bsidxs = np.random.randint(n, size=[n, n])
            # bsidxs = np.stack([np.random.choice(n, size=n, replace=False)
            #          for _ in range(n)])
            bsidxs = np.take(T*np.stack(list(idxs.values())),bsidxs, axis=1).reshape(m*n, -1)
            avg_idxs = np.repeat(np.take(np.stack(d['seed']), bsidxs), T, axis=0)
            bsidxs = np.repeat(bsidxs, T, axis=0) + np.tile(np.arange(T), len(bsidxs))[:, None]

            # all_idxs = d.groupby(groupby).indices
            # configs = np.tile(np.stack(list(all_idxs.keys())), (n, 1))
            configs = np.take(np.stack(d[groupby].values), bsidxs, axis=0)[:, 0, :]
            avg = pd.DataFrame(configs, columns=groupby)
            data = {'avg_idxs': [*avg_idxs]}
            for k in keys:
                yhs = np.take(np.stack(d[k]), bsidxs, axis=0).mean(1)
                data[k] = [*yhs]
            avg = avg.assign(**data)
            avg['avg_idxs'] = avg.apply(lambda r: str(r['avg_idxs']), axis=1)
        else:
            avg = d.groupby(groupby)[keys].mean(numeric_only=False).reset_index()

        for k in keys:
            avg[k] = avg.apply(lambda r: r[k].reshape(n_data[k], -1), axis=1)
            d[k] = d.apply(lambda r: r[k].reshape(n_data[k], -1), axis=1)

    if get_err:
        for k in keys:
            ykey = 'train' if k == 'yh' else 'val'
            y = th.tensor(get_data()[ykey].targets).long().to(dev)
            n = len(y)
            out = np.stack(avg[k])
            preds = np.argmax(out, -1)
            err = ((th.tensor(preds).to(dev) != y).sum(1) / n).cpu().numpy()
            avg[f'{ykey}_err'] = err


    if compute_distance:
        dists = []
        indices = d.groupby(groupby).indices
        for i in range(len(avg)):
            config = tuple(avg.iloc[i][groupby])
            ii = indices[config]
            for k in keys:
                x1 = th.Tensor(avg.iloc[i][k]).unsqueeze(0).transpose(0, 1).to(dev)
                x2 = th.Tensor(np.stack(d.iloc[ii][k])).transpose(0, 1).to(dev)
                dist = distf(x2, x1)
                for (j, dj) in enumerate(dist):
                    dic = dict(dist=dj.item(), key=k)
                    dic.update({groupby[i]:config[i] for i in range(len(groupby))})
                    dists.append(dic)
        dists = pd.DataFrame(dists)
        d_return['dists'] = dists

    if update_d:
        avg['avg'] = True
        avg['seed'] = -1
        d['avg'] = False
        if bootstrap:
            d['avg_idxs'] = pd.Series([(-1, 0) for _ in range(len(d))])
        d = pd.concat([avg, d]).reset_index()
        d_return['d'] = d
    else:
        d_return['avg'] = avg

    return d_return


def interpolate(d, ts, pts, columns=['seed', 'm', 'opt', 'avg'], keys=['yh', 'yvh'], dev='cuda'):
    r = []
    N = len(pts)
    y = get_data()
    y = {'y': th.tensor(y['train'].targets).long().to(dev), 
         'yh': th.tensor(y['val'].targets).long().to(dev)}
    configs = d.groupby(columns).indices
    for (c, idx) in configs.items():
        traj_interp = {}
        for k in keys:
            traj = np.stack(d.iloc[idx][k])
            traj_ = interpn(ts.reshape(1, -1), traj, pts)
            traj_interp[k] = th.Tensor(traj_).to(dev)
        for i in range(N):
            t = {}
            t.update({columns[i]: ci for (i, ci) in enumerate(c)})
            t.update({'t': pts[i]})
            for k in keys:
                ti = traj_interp[k][i, :].reshape(-1, 10)
                t.update({k: ti.cpu().numpy()})
                f = -th.gather(th.log(ti+1e-8), 1, y[k[:-1]].view(-1, 1)).mean()
                acc = th.eq(th.argmax(ti, axis=1), y[k[:-1]]).float()
                if k == 'yh':
                    t.update({'favg': f.mean().item(), 'err': 1-acc.mean().item()})
                else:
                    t.update({'vfavg': f.mean().item(),
                            'verr': 1-acc.mean().item()})
            r.append(t)
    d = pd.DataFrame(r)
    return d
