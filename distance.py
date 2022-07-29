from ensurepip import bootstrap
import os
import argparse
import torch as th
import numpy as np
from itertools import product
from functools import partial
import tqdm
from utils import load_d, avg_model, interpolate


def dbhat(x1, x2, reduction='mean', dev='cuda', debug=False, chunks=0):
    # x1, x2 shape (num_points, num_samples, num_classes)
    np1, ns, _ = x1.size()
    np2, ns, _ = x2.size()
    print(np1, np2, ns)
    x1, x2 = x1.transpose(0, 1), x2.transpose(0, 1)
    w = np.zeros([np1, np2])
    if debug:
        assert th.allclose(x1.sum(-1), th.ones(ns, np1)) and th.allclose(x2.sum(-1), th.ones(ns, np2))
    chunks = chunks or 1
    for aa in tqdm.tqdm(th.chunk(th.arange(ns), chunks)):
        xx1 = x1[aa, :].to(dev)
        xx2 = x2[aa, :].to(dev)
        aa = th.sqrt(aa)
        w_ = -th.log(th.bmm(th.sqrt(xx1), th.sqrt(xx2).transpose(1, 2)))
        w_[w_ == th.inf] = 100
        w_[w_ < 0] = 0
        if th.isnan(w_).sum()>0:
            import ipdb; ipdb.set_trace()
        w += w_.sum(0).cpu().numpy()
    if reduction == 'mean':
        return w / ns
    else:
        return w


def diskl(x1, x2, reduction='mean', probs=True, dev='cuda', chunks=0):
    np1, ns, _ = x1.size()
    np2, ns, _ = x2.size()
    x1, x2 = x1.transpose(0, 1).to(dev), x2.transpose(0, 1).to(dev)
    w = np.zeros([np1, np2])
    chunks = chunks or 1
    for aa in th.chunk(th.arange(ns), chunks):
        xx1 = x1[aa, :].to(dev)
        xx2 = x2[aa, :].to(dev)
        if probs:
            logxx1, logxx2 = th.log(xx1), th.log(xx2)
            logxx1[logxx1 == -th.inf] = 100
            logxx2[logxx2 == -th.inf] = 100
        else:
            logxx1, logxx2 = xx1, xx2
            xx1, xx2 = th.exp(xx1), th.exp(xx2)
        w12 = (xx1 * logxx2).sum(-1, keepdim=True) - th.bmm(xx1, logxx2.transpose(1, 2))
        w21 = (xx1 * logxx2).sum(-1, keepdim=True) - th.bmm(logxx1, xx2.transpose(1, 2))
        w += 0.5*(w12+w21)
    if reduction == 'mean':
        return w/ns
    else:
        return w



def dinpca(x1, x2, sign=1, dev='cuda', sqrt=False):
    # x1, x2  shape (nmodels, ncoords)
    # sign (ncoords, ), sign of each coordinate
    x1, x2, sign = x1.to(dev), x2.to(dev), sign.to(dev)
    d = (((x1[None, ...] - x2[:, None, ...])**2) * sign.reshape(1, 1, -1)).sum(-1)
    if sqrt:
        return th.sqrt(th.maximum(d, th.zeros_like(d)))
    return d


def dfrechet(x1, x2, distf=partial(dbhat, reduction='mean')):
    # code adapted from https://github.com/spiros/discrete_frechet/blob/master/frechetdist.py
    d = len(x1)
    D = distf(x1, x2)
    ca = (th.ones((d, d)) * -1)
    dist = _c(ca, d-1, d-1, D)
    return dist


def _c(ca, i, j, D):
    # helper function for dfrechet
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = D[i, j]
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i-1, 0, D), D[i, j])
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j-1, D), D[i, j])
    elif i > 0 and j > 0:
        ca[i, j] = max(
            min(
                _c(ca, i-1, j, D),
                _c(ca, i-1, j-1, D),
                _c(ca, i, j-1, D)
            ),
            D[i, j]
        )
    else:
        ca[i, j] = th.inf 
    return ca[i, j]


def dp2t(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None, return_idxs=False):
    # xs: (npoints, num_samples, num_classes)
    # Y: (T, num_samples, num_classes)
    pdists = dbhat(xs, Y, reduction, dev)
    if s == 0.0:
        kdist, idxs = pdists.min(1)
        if return_idxs:
            return kdist, idxs
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


def pairwise_dist(d, groups=['m', 'opt', 'seed'], k='yh', distf=dt2t, **kwargs):
    groups = d.groupby(groups).indices
    configs = list(groups.keys())
    dists = np.zeros([len(configs), len(configs)])
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            c1, c2 = configs[i], configs[j]
            x1 = th.Tensor(np.stack(d.iloc[groups[c1]][k].values))
            x2 = th.Tensor(np.stack(d.iloc[groups[c2]][k].values))
            dist = distf(x1, x2, **kwargs)
            if isinstance(dist, dict):
                dists[i, j], dists[j, i] = dist.items()
            else:
                dists[i, j] = dist
    return dists, configs


def dp2t_batch(xs, Y, reduction='mean', dev='cuda', s=0.1, dys=None, return_idxs=False):
    # xs: (npoints, num_samples, num_classes)
    # Y: (N, T, num_samples, num_classes)
    N, T = Y.shape[:2]
    Y = Y.flatten(0, 1)
    pdists = dbhat(xs, Y, reduction, dev)
    pdists = th.stack(th.split(pdists, T, 1), dim=1)
    if s == 0.0:
        kdist, idxs = pdists.min(-1)
        if return_idxs:
            return kdist, idxs
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
            if sym == 'none':
                distxy, distyx = dt2t_batch(th.Tensor(x1), th.Tensor(x2), reduction='mean', dev=dev, s=s, sym=sym, normalization=normalization).values()
                row, col = zip(*list(product(c1, c2)))
                dists[row, col] = distxy.flatten().cpu().numpy()
                dists.T[row, col] = distyx.flatten().cpu().numpy()
            else:
                dist = dt2t_batch(th.Tensor(x1), th.Tensor(x2), reduction='mean', dev=dev, s=s, sym=sym, normalization=normalization)
                row, col = zip(*list(product(c1, c2)))
                dists[row, col] = dist.flatten().cpu().numpy()
            print(dists[:4, :4])
                
    return dists, configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=float, default=0.0)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--sym', type=str, default='mean')
    parser.add_argument('--norm', type=str, default='length')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--interpolate', action='store_true')
    args = parser.parse_args()


    loc = 'results/models/new'
    fname = f'dist_{args.s}_{args.sym}_{args.norm}'
    print(fname)
    if args.bootstrap:
        fname += '_bootstrap'
    if args.interpolate:
        fname += '_interpolate'

    varying = {
        "m": ["wr-4-8", "allcnn-96-144", "fc-1024-512-256-128"],
        "opt": ["adam", "sgdn", "sgd"]
    }
    fixed = {"aug": [True],
             "wd": [0.0],
             "bn": [True],
             "bs": [200]}

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

    d = avg_model(d, groupby=list(varying.keys()) + ['t'], probs=True, get_err=True, bootstrap=args.bootstrap,
                update_d=True, compute_distance=False, dev='cuda')['d']
    columns = list(varying.keys()) + ['seed', 'avg']
    if args.bootstrap:
        columns += ['avg_idxs']
    if args.interpolate:
        d = interpolate(d, ts, pts, columns=columns, keys=['yh'], dev='cuda')

    print("computing pairwise distance")
    groups = list(varying.keys()) + ['seed'] # avg==True iff seed==-1
    if args.bootstrap:
        groups += ['avg_idxs']

    # dists, configs = pairwise_dist(d, groups=groups, distf=dfrechet)
    dists, configs = pairwise_dist_batch(d, groups=groups, s=args.s, batch=args.batch, sym=args.sym, normalization=args.norm)
    symd = np.copy(dists)
    symd[np.tril_indices(len(dists), -1)] = 0
    symd = symd+ symd.T

    th.save({'dists': dists, 'symd': symd, 'configs': configs, 'groups': list(varying.keys()) + ['seed']},
            os.path.join(loc, f'{fname}.p'))
