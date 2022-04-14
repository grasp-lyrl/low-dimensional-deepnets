import torch as th

def dbhat(x1, x2, reduction='mean', dev='cuda', debug=False):
    # x1, x2 shape (num_models, num_samples, num_classes)
    nm, ns, _ = x1.size()
    x1, x2 = x1.transpose(0, 1).to(dev), x2.transpose(0, 1).to(dev)
    if debug:
        assert th.allclose(x1.sum(-1), th.ones(ns, nm).to(dev)) and th.allclose(x2.sum(-1), th.ones(ns, nm).to(dev))
    return getattr(-th.log(th.bmm(th.sqrt(x1), th.sqrt(x2).transpose(1, 2))), reduction)(0)


def dinpca(x1, x2, sign):
    # x1, x2  shape (nmodels, ncoords)
    # sign (nmodels, ), sign of each coordinate
    dinpca = ((x1[..., None] - x2[..., None, :])**2 * sign.reshape(-1, 1, 1)).sum(0)
    return dinpca

def traj_dbhat(X1, X2, reduction='mean', dev='cuda', s=0.1):
    # X1, X2 shape (num_models, T, num_samples, num_classes)
    X1, X2 = X1.flatten(0, 2), X2.flatten(0, 2)
    dists = dbhat(X1, X2, reduction=reduction, dev=dev)
