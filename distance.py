import torch as th

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
    pdists = dbhat(xs, Y, reduction, dev)[:, :-1]
    if dys is None:
        dys = th.sqrt(th.diag(dbhat(Y, Y, reduction, dev), 1))
    Z = th.exp(-pdists/(2*s**2)).sum(1)
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


