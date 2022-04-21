import torch as th
import numpy as np
from itertools import product
from utils import *


def dbhat(x1, x2, reduction='mean', dev='cuda', debug=False):
    # x1, x2 shape (num_models, num_samples, num_classes)
    nm, ns, _ = x1.size()
    x1, x2 = x1.transpose(0, 1).to(dev), x2.transpose(0, 1).to(dev)
    if debug:
        assert th.allclose(x1.sum(-1), th.ones(ns, nm).to(dev)) and th.allclose(x2.sum(-1), th.ones(ns, nm).to(dev))
    return getattr(-th.log(th.bmm(th.sqrt(x1), th.sqrt(x2).transpose(1, 2))), reduction)(0)


def dinpca(x1, x2, sign=1, dev='cuda', sqrt=False):
    # x1, x2  shape (nmodels, ncoords)
    # sign (ncoords, ), sign of each coordinate
    x1, x2, sign = x1.to(dev), x2.to(dev), sign.to(dev)
    d = (((x1[None, ...] - x2[:, None, ...])**2) * sign.reshape(1, 1, -1)).sum(-1)
    if sqrt:
        return th.sqrt(th.maximum(d, th.zeros_like(d)))
    return d


def dp2t(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None, use_min=False):
    # xs: (npoints, num_samples, num_classes)
    # Y: (T, num_samples, num_classes)
    pdists = dbhat(xs, Y, reduction, dev)
    if use_min:
        kdist, _ = pdists.min(1)
    else:
        if dys is None:
            dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
        pdists = pdists[:, :-1]
        Z = (th.exp(-pdists/(2*s**2)) * dys).sum(1)
        kdist = (th.exp(-pdists/(2*s**2)) * pdists * dys.reshape(1, -1)).sum(1) / Z
    return kdist


def dt2t(X, Y, reduction='mean',  dev='cuda',s=0.1, sym=False):
    # X: (T, num_samples, num_classes)
    # Y: (T, num_samples, num_classes)
    dxs = th.sqrt(th.diag(dbhat(X, X, reduction, dev), 1))
    dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
    dxy = (dp2t(X, Y, reduction, dev, s, dys)[:-1] * dxs).sum() 
    dyx = (dp2t(Y, X, reduction, dev, s, dxs)[:-1] * dys).sum() 
    if sym:
        return (dxy+dyx).item()/2
    return dxy.item(), dyx.item()


def dp2t_batch(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None, use_min=False):
    # xs: (npoints, num_samples, num_classes)
    # Y: (N, T, num_samples, num_classes)
    N, T = Y.shape[:2]
    Y = Y.flatten(0, 1)
    pdists = dbhat(xs, Y, reduction, dev)
    pdists = th.stack(th.split(pdists, T, 1), dim=1)
    if use_min:
        kdist, _ = pdists.min(-1)
    else:
        if dys is None:
            dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
            dys = th.stack([dys[i*T:(i+1)*T-1] for i in range(N)]).unsqueeze(0)
        pdists = pdists[:, :, :-1]
        Z = (th.exp(-pdists/(2*s**2)) * dys).sum(-1)
        kdist = (th.exp(-pdists/(2*s**2)) * pdists * dys).sum(-1) / Z
    return kdist

def dt2t_batch(X, Y, reduction='mean',  dev='cuda', s=0.1, use_min=False):
    # X: (Nx, Tx, num_samples, num_classes)
    # Y: (Ny, Ty, num_samples, num_classes)
    Nx, Tx = X.shape[:2]
    Ny, Ty = Y.shape[:2]
    x, y = X.flatten(0, 1), Y.flatten(0, 1)

    dxs = th.sqrt(th.diag(dbhat(x, x, reduction, dev), 1))
    dxs = th.stack([dxs[i*Tx:(i+1)*Tx-1] for i in range(Nx)])

    dys = th.sqrt(th.diag(dbhat(y, y, reduction, dev), 1))
    dys = th.stack([dys[i*Ty:(i+1)*Ty-1] for i in range(Ny)])

    dxy = th.stack(th.split(dp2t_batch(xs=x, Y=Y, reduction=reduction, dev=dev, s=s, dys=dys.unsqueeze(0), use_min=use_min), Tx, 0), dim=0)
    dxy = (dxy[:, :-1, :] * dxs.unsqueeze(-1)).sum(1)
    dyx = th.stack(th.split(dp2t_batch(xs=y, Y=X, reduction=reduction, dev=dev, s=s, dys=dxs.unsqueeze(0), use_min=use_min), Ty, 0), dim=0)
    dyx = (dyx[:, :-1, :] * dys.unsqueeze(-1)).sum(1)
    return (th.sqrt(dxy)+th.sqrt(dyx).T)/2


def pairwise_dist(d, groupby=['m', 'opt', 'seed'], s=0.1, k='yh', use_min=False):
    groups = d.groupby(groupby).indices
    configs = list(groups.keys())
    dists = np.zeros([len(configs), len(configs)])
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            c1, c2 = configs[i], configs[j]
            x1 = th.Tensor(np.stack(d.iloc[groups[c1]][k].values))
            x2 = th.Tensor(np.stack(d.iloc[groups[c2]][k].values))
            dists[i, j], dists[j, i] = dt2t(x1, x2, s=s, use_min=use_min)
    return dists, configs


def pairwise_dist_batch(d, groups=['m', 'opt', 'seed'], dev='cuda', s=0.1, k='yh', use_min=False, batch=10):
    groups = d.groupby(groups).indices
    configs = list(groups.keys())
    dists = np.zeros([len(configs), len(configs)])
    bidxs = [np.arange(len(configs))[i*batch:(i+1)*batch]
             for i in range(len(configs)//batch+1)]
    for i in range(len(bidxs)):
        for j in range(i, len(bidxs)):
            c1, c2 = bidxs[i], bidxs[j]
            if len(c1) == 0 or len(c2) == 0:
                continue
            print(c1, c2)
            x1 = np.stack([np.stack(d.iloc[groups[configs[i]]][k])
                        for i in c1])
            x2 = np.stack([np.stack(d.iloc[groups[configs[i]]][k])
                        for i in c2])
            dist = dt2t_batch(th.Tensor(x1), th.Tensor(x2), reduction='mean', dev=dev, s=s, use_min=use_min)
            row, col = zip(*list(product(c1, c2)))
            dists[row, col] = dist.flatten().cpu().numpy()
    return dists, configs


if __name__ == "__main__":
    s = 0.1
    batch = 2
    loc = 'results/models/new'
    use_min = True 
    fname = 'pairwise_dists_bs_mdist'

    varying = {
        "bs": [200, 400],
        "m": ["wr-4-8", "allcnn-96-144", "fc-1024-512-256-128"],
        "opt": ["adam", "sgdn", "sgd"]
    }
    fixed = {"aug": [True],
             "wd": [0.0],
             "bn": [True]}

    T = 45000
    ts = []
    for t in range(T):
        if t < T//10:
            if t % (T//100) == 0:
                ts.append(t)
        else:
            if t % (T//10) == 0 or (t == T-1):
                ts.append(t)
    ts = np.array(ts)
    pts = np.concatenate(
        [np.arange(ts[i], ts[i+1], (ts[i+1]-ts[i]) // 5) for i in range(len(ts)-1)])

    print("loading model")
    d = load_d(loc, cond={**varying, **fixed}, avg_err=True, drop=False, probs=True)

    d = avg_model(d, groupby=list(varying.keys()) + ['t'], probs=True, get_err=True,
                update_d=True, compute_distance=False, dev='cuda')['d']
    d = interpolate(d, ts, pts, columns=list(varying.keys()) + ['seed', 'avg'], keys=['yh'], dev='cuda')

    print("computing pairwise distance")
    dists, configs = pairwise_dist_batch(d, groups=list(varying.keys()) + ['seed'], use_min=use_min, s=s, batch=batch)
    symd = dists
    symd[np.tril_indices(len(dists), -1)] = 0
    symd = symd+ symd.T

    th.save({'dists': dists, 'symd': symd, 'configs': configs, 'groups': list(varying.keys()) + ['seed']},
            os.path.join(loc, f'{fname}.p'))
