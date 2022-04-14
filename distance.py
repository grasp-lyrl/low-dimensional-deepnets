import torch as th

def dbhat(x1, x2, reduction='mean', dev='cuda', debug=False):
    # x1, x2 torch.Tensor of shape (num_samples, num_models, num_classes)
    ns, nm, nc = x1.size()
    x1, x2 = x1.to(dev), x2.to(dev)
    if debug:
        assert th.allclose(x1.sum(-1), th.ones(ns, nm)) and th.allclose(x2.sum(-1), th.ones(nm, ns))
    return getattr(-th.log(th.bmm(th.sqrt(x1), th.sqrt(x2).transpose(1, 2))), reduction)(0)


def pairwise_d(df, distf=dbhat, dev='cuda'):