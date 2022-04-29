import os
import torch as th
import numpy as np
from itertools import product
from utils import load_d, avg_model, interpolate


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


def dp2t(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None):
    # xs: (npoints, num_samples, num_classes)
    # Y: (T, num_samples, num_classes)
    pdists = dbhat(xs, Y, reduction, dev)
    if s == 0.0:
        kdist, _ = pdists.min(1)
    else:
        if dys is None:
            dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
        pdists = pdists[:, :-1]
        Z = (th.exp(-pdists/(2*s**2)) * dys).sum(1)
        kdist = (th.exp(-pdists/(2*s**2)) * pdists * dys.reshape(1, -1)).sum(1) / Z
    return kdist


def dt2t(X, Y, reduction='mean', dev='cuda', s=0.1, sym='min', normalization='length'):
    # X: (T, num_samples, num_classes)
    # Y: (T, num_samples, num_classes)
    dxs = th.sqrt(th.diag(dbhat(X, X, reduction, dev), 1))
    dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
    lx = dxs.sum()
    ly = dys.sum()
    dxy = (dp2t(X, Y, reduction, dev, s, dys)[:-1] * dxs).sum() 
    dyx = (dp2t(Y, X, reduction, dev, s, dxs)[:-1] * dys).sum() 

    if normalization=='length':
        dxy /= lx
        dyx /= ly

    if sym == 'min':
        return min(dxy.item(), dyx.item())
    elif sym == 'mean':
        return (dxy+dyx).item()/2
    elif sym == 'min_length':
        return dxy.item() if lx < ly else dyx.item()
    else:
        return {'dxy': dxy.item(), 'dyx': dyx.item()}


def pairwise_dist(d, groupby=['m', 'opt', 'seed'], s=0.1, k='yh', sym='min', normalization='length', dev='cuda', reduction='mean'):
    groups = d.groupby(groupby).indices
    configs = list(groups.keys())
    dists = np.zeros([len(configs), len(configs)])
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            c1, c2 = configs[i], configs[j]
            x1 = th.Tensor(np.stack(d.iloc[groups[c1]][k].values))
            x2 = th.Tensor(np.stack(d.iloc[groups[c2]][k].values))
            dists[i, j], dists[j, i] = dt2t(x1, x2, reduction=reduction, dev=dev, s=s, sym=sym, normalization=normalization)
    return dists, configs


def dp2t_batch(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None):
    # xs: (npoints, num_samples, num_classes)
    # Y: (N, T, num_samples, num_classes)
    N, T = Y.shape[:2]
    Y = Y.flatten(0, 1)
    pdists = dbhat(xs, Y, reduction, dev)
    pdists = th.stack(th.split(pdists, T, 1), dim=1)
    if s == 0.0:
        kdist, _ = pdists.min(-1)
    else:
        if dys is None:
            dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
            dys = th.stack([dys[i*T:(i+1)*T-1] for i in range(N)]).unsqueeze(0)
        pdists = pdists[:, :, :-1]
        Z = (th.exp(-pdists/(2*s**2)) * dys).sum(-1)
        kdist = (th.exp(-pdists/(2*s**2)) * pdists * dys).sum(-1) / Z
    return kdist


def dt2t_batch(X, Y, reduction='mean', dev='cuda', s=0.1, sym='min', normalization='length'):
    # X: (Nx, Tx, num_samples, num_classes)
    # Y: (Ny, Ty, num_samples, num_classes)
    Nx, Tx = X.shape[:2]
    Ny, Ty = Y.shape[:2]
    x, y = X.flatten(0, 1), Y.flatten(0, 1)

    dxs = th.sqrt(th.diag(dbhat(x, x, reduction, dev), 1))
    dxs = th.stack([dxs[i*Tx:(i+1)*Tx-1] for i in range(Nx)])
    lx = dxs.sum(-1, keepdim=True)

    dys = th.sqrt(th.diag(dbhat(y, y, reduction, dev), 1))
    dys = th.stack([dys[i*Ty:(i+1)*Ty-1] for i in range(Ny)])
    ly = dys.sum(-1, keepdim=True)

    dxy = th.stack(th.split(dp2t_batch(xs=x, Y=Y, reduction=reduction, dev=dev, s=s, dys=dys.unsqueeze(0)), Tx, 0), dim=0)
    dxy = (dxy[:, :-1, :] * dxs.unsqueeze(-1)).sum(1)
    dyx = th.stack(th.split(dp2t_batch(xs=y, Y=X, reduction=reduction, dev=dev, s=s, dys=dxs.unsqueeze(0)), Ty, 0), dim=0)
    dyx = (dyx[:, :-1, :] * dys.unsqueeze(-1)).sum(1)

    if normalization == 'length':
        dxy /= lx
        dyx /= ly

    if sym == 'min':
        return th.minimum(dxy, dyx.T)
    elif sym == 'mean':
        return (dxy+dyx.T)/2
    elif sym == 'min_length':
        mask = lx.repeat(1, Nx) < ly.T.repeat(Ny, 1)
        dyx[mask] = dxy[mask]
        return dyx
    else:
        return {'dxy': dxy, 'dyx': dyx}


def pairwise_dist_batch(d, groups=['m', 'opt', 'seed'], dev='cuda', s=0.1, k='yh', batch=10, sym='min', normalization='length'):
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
            dist = dt2t_batch(th.Tensor(x1), th.Tensor(x2), reduction='mean', dev=dev, s=s, sym=sym, normalization=normalization)
            row, col = zip(*list(product(c1, c2)))
            dists[row, col] = dist.flatten().cpu().numpy()
    return dists, configs


if __name__ == "__main__":
    s = 0.1
    batch = 2 
    symmetrize = 'min'
    normalization = 'length'
    loc = 'results/models/new'
    fname = f'dist_{s}_{symmetrize}_{normalization}'

    varying = {
        "bs": [200],
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
    dists, configs = pairwise_dist_batch(d, groups=list(varying.keys()) + ['seed'], s=s, batch=batch, sym=symmetrize, normalization=normalization)
    symd = dists
    symd[np.tril_indices(len(dists), -1)] = 0
    symd = symd+ symd.T

    th.save({'dists': dists, 'symd': symd, 'configs': configs, 'groups': list(varying.keys()) + ['seed']},
            os.path.join(loc, f'{fname}.p'))
