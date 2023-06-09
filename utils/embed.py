import numpy as np
import scipy.linalg as sp
import torch as th

import os
import h5py
import pandas as pd

from utils import *
from utils import distance


def xembed(
    d1,
    d2=None,
    extra_pts=None,
    fn="",
    key="yh",
    loc="inpca_results",
    ss=slice(0, None, 1),
    probs=False,
    ne=3,
    force=False,
    idx=None,
    dev="cuda",
    distf="dbhat",
    reduction="sum",
    chunks=1,
    proj=False,
    save_didx=False,
):
    idx = idx or ["seed", "widen", "numc", "t", "err", "verr", "favg", "vfavg"]
    d2 = d1 if d2 is None else d2
    
    dr = d1.reindex(list(set(d1.columns).union(set(idx))), axis=1)[idx]
    dc = d2.reindex(list(set(d1.columns).union(set(idx))), axis=1)[idx] if d2 is not None else dr
    if extra_pts is not None:
        qc = extra_pts.loc[:, extra_pts.columns.isin(idx)]
        dc = pd.concat([dc, qc])
        q = th.Tensor(
            np.stack([extra_pts.iloc[i][key][ss] for i in range(len(extra_pts))])
        ).cpu()
    if save_didx:
        th.save({"dr": dr, "dc": dc}, os.path.join(loc, "didx_%s.p" % fn))
    n, m = len(dr), len(dc)  # number of models

    x = th.Tensor(np.stack([d1.iloc[i][key].squeeze()[ss] for i in range(n)]).squeeze()).cpu()
    if d2 is None:
        y = th.Tensor(np.stack([d1.iloc[i][key].squeeze()[ss] for i in range(n)]).squeeze()).cpu()
    else:
        y = th.Tensor(np.stack([d2.iloc[i][key].squeeze()[ss] for i in range(m)]).squeeze()).cpu()

    if (not os.path.isfile(os.path.join(loc, "w_%s.p" % fn))) or force:
        if "kl" in distf:
            w = getattr(distance, distf)(
                x, y, reduction=reduction, dev=dev, chunks=chunks, probs=probs
            )
        else:
            x = th.exp(x) if not probs else x
            y = th.exp(y) if not probs else y
            w = getattr(distance, distf)(
                x, y, reduction=reduction, dev=dev, chunks=chunks
            )
        print("Saving w")
        th.save(w, os.path.join(loc, "w_%s.p" % fn), pickle_protocol=4)
    else:
        print("Found: ", os.path.join(loc, "w_%s.p" % fn))
        w = th.load(os.path.join(loc, "w_%s.p" % fn))

    if proj:
        d_mean = w.mean(0)
        l = np.eye(w.shape[0]) - 1.0 / w.shape[0]
        w = -l @ w @ l / 2
        r = proj_(w, n, ne)
        if extra_pts is not None:
            q = lazy_embed(
                q,
                x,
                w,
                d_mean,
                evals=r["e"],
                evecs=r["v"],
                distf=distf,
                ne=ne,
                chunks=chunks,
            )
            r["xp"] = np.vstack([r["xp"], q])
        th.save(r, os.path.join(loc, "r_%s.p" % fn))
    return


# Calculate the embedding of a distibution q in the intensive embedding of models ps with divergence=distf, supply d_list the precalculated matrix of distances pf p_list.
def lazy_embed(
    new_pts=None,
    ps=None,
    w=None,
    d_mean=None,
    dp=None,
    evals=None,
    evecs=None,
    distf="dbhat",
    ne=3,
    chunks=1,
    dev='cuda',
):
    # w: centered pairwise distance, d_mean: mean before centering
    if dp is None:
        dp = getattr(distance, distf)(new_pts, ps, chunks=chunks, dev=dev)
    d_mean_mean = np.mean(d_mean)
    if (evals is None) or (evecs is None):
        N, _, _ = ps.shape
        _, _, evals, evecs = proj_(w, N, ne).values()
    dp_mean = dp - np.mean(dp, 1, keepdims=True) - d_mean + d_mean_mean
    dp_mean = -0.5 * dp_mean
    sqrtsigma = np.sign(evals)*np.sqrt(np.abs(evals))
    return (1 / sqrtsigma) * np.matmul(dp_mean, evecs)


def full_embed(dists):
    dmean = dists.mean(0)
    l = np.eye(dists.shape[0]) - 1.0 / dists.shape[0]
    dists = -l @ dists @ l / 2
    e, v = np.linalg.eigh(dists)
    ii = np.argsort(np.abs(e))[::-1]
    e, v = e[ii], v[:, ii]
    xp = v * np.sqrt(np.abs(e))
    r = dict(xp=xp, e=e, v=v)
    return dmean, r


def proj_(w, n, ne):
    print("Projecting")
    e1, v1 = sp.eigh(
        w, driver="evx", check_finite=False, subset_by_index=[n - (ne + 1), n - 1]
    )
    e2, v2 = sp.eigh(w, driver="evx", check_finite=False, subset_by_index=[0, (ne + 1)])
    e = np.concatenate((e1, e2))
    v = np.concatenate((v1, v2), axis=1)

    ii = np.argsort(np.abs(e))[::-1]
    e, v = e[ii], v[:, ii]
    xp = v * np.sqrt(np.abs(e))
    return dict(xp=xp, e=e, v=v)


def explained_distance(r):
    ii = np.argsort(np.abs(r['es']))[::-1]
    es = r['es'][ii]
    vs = r['vs'][:, ii]
    b = r['diag']
    n = len(b)
    trb = np.sum(b)
    dfnorm = 2*trb**2 + 2*n*np.linalg.norm(b)**2+4*r['fn']**2

    df = []
    for i in range(len(es)):
        t1 = 2*(trb - es[:i].sum())**2
        bk = ((vs[:, :i]**2) * es[:i]).sum(-1)
        t2 = 2*n*np.linalg.norm(b - bk)**2
        t3 = 4*(r['fn']**2 - (es[:i]**2).sum())
        df.append(np.sqrt((t1+t2+t3) / dfnorm))
    return df


def explained_stress(r):
    ii = np.argsort(np.abs(r['es']))[::-1]
    es = r['es'][ii][:50]
    return 1 - np.sqrt(1-np.cumsum(es ** 2)/r['fn']**2)


def weighted_MDS(w, weight, ne=3):

    dmean = (w*weight).sum(1, keepdims=True)
    w = w-dmean
    w = w-(w*weight[:, None]).sum(0)
    w0 = -0.5*w

    w = w0 * weight[None, :]
    e, v = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)
    ii = np.argsort(np.abs(e))[::-1]
    e, v = e[ii], v[:, ii]

    scaling_factor = np.linalg.norm(
        v*np.sqrt(weight[:, None]), axis=0, keepdims=True)
    print(scaling_factor)
    xp = v*np.sqrt(np.abs(e)) / scaling_factor
    r = dict(xp=xp, e=e, v=v, dmean=dmean)
    return r


def project(
    cond={"seed": [45]},
    save_loc="45",
    fn="yh_all",
    err_threshold=0.1,
    extra_points=None,
    force=False,
    save_w=False,
):
    # select sub-matrix from the entire distance matrix and compute projections
    loc = "inpca_results_all"
    ne = 5

    groups = ["seed", "m", "opt", "bs", "aug", "lr", "wd"]
    folder = os.path.join(loc, save_loc)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.isfile(os.path.join(folder, f"didx_{fn}.p")) and not force:
        idx, didx = th.load(os.path.join(folder, f"didx_{fn}.p"))
    else:
        # filter out un-trained model
        idx = []
        didx = th.load(os.path.join(loc, f"didx_yh_all.p"))
        for (c, indices) in didx.groupby(groups).indices.items():
            well_trained = didx.iloc[indices[-1]]["err"] < err_threshold
            meet_conds = True
            for (i, g) in enumerate(groups):
                meet_conds = meet_conds and (
                    (c[i] in cond[g]) if g in cond.keys() else True
                )
            if (well_trained and meet_conds) or c[0] == 0:
                idx.extend(indices)
        idx = sorted(idx)
        didx = didx.iloc[idx]
        th.save((idx, didx), os.path.join(folder, f"didx_{fn}.p"))

    print(len(idx))
    f = h5py.File(os.path.join(loc, f"w_{fn}.h5"), "r")
    w = f["w"][:, idx][idx, :]
    n = w.shape[0]
    d_mean = w.mean(0)

    if save_w:
        th.save(w, os.path.join(folder, f"w_{fn}.p"))
    if os.path.isfile(os.path.join(folder, f"r_{fn}.p")):
        r = th.load(os.path.join(folder, f"r_{fn}.p"))
    else:
        l = np.eye(w.shape[0]) - 1.0 / w.shape[0]
        w = -l @ w @ l / 2
        r = proj_(w, n, ne)

    if extra_points is not None:
        didx_ = th.load(os.path.join(loc, f"didx_{fn}.p"))
        ridx = get_idx(didx_, extra_points)
        dp = f["w"][:, idx][ridx, :]
        q = lazy_embed(dp=dp, d_mean=d_mean, evals=r["e"], evecs=r["v"], ne=ne)
        r["xp"] = np.vstack([r["xp"], q])
        didx = pd.concat([didx, didx_.iloc[ridx]])
        th.save((idx + ridx, didx), os.path.join(folder, f"didx_{fn}_all.p"))

    th.save(r, os.path.join(folder, f"r_{fn}.p"), pickle_protocol=4)


if __name__ == "__main__":
    # cond = {"seed":[43, 46], "aug": ["simple"]}
    project(
        cond={
            "m": [
                "allcnn",
                "convmixer",
                "vit",
                "wr-10-4-8",
                "wr-16-4-64",
                "true",
                "random",
            ],
            "aug": ["simple"],
        },
        save_loc="no_fc",
        fn="yvh_all",
        err_threshold=0.1,
        extra_points=None,
        force=True,
        save_w=False,
    )
    # root_dir = "/home/ubuntu/ext_vol/inpca/results/models/all"
    # conds = [{"bseed": -1, "aug": "none", "m": "allcnn", "bn": True, "drop": 0.0, "opt": "sgdn", "bs": 200, "lr": 0.1, "wd": 0.0, "corner":"subsample-200"},
    # {"bseed":-1,"aug":"none","m":"fc","bn":True,"drop":0.0,"opt":"sgdn","bs":200,"lr":0.1,"wd":0.0,"corner":"subsample-2000"},
    # {"bseed":-1,"aug":"none","m":"wr-10-4-8","bn":True,"drop":0.0,"opt":"sgdn","bs":200,"lr":0.1,"wd":0.0,"corner":"normal"},
    # {"bseed":-1,"aug":"none","m":"wr-10-4-8","bn":True,"drop":0.0,"opt":"sgdn","bs":200,"lr":0.1,"wd":0.0,"corner":"uniform"}
    # ]
    # flist = []
    # for seed in range(42,52):
    #     for c in conds:
    #         c_ = {"seed":seed}
    #         c_.update(c)
    #         flist.append(os.path.join(root_dir, f"{json.dumps(c_).replace(' ', '')}.p"))

    # d = load_d(file_list=flist, avg_err=True, verbose=False,
    #            probs=True, numpy=True, return_nan=False, loaded=False).reset_index(drop=True)

    # idx = ['seed', 'aug', 'm', 'bn', 'opt', 'bs', 'lr',
    #    'wd', 'corner', 't', 'err',
    #    'verr', 'favg', 'vfavg']
    # xembed(d, d2=None, extra_pts=None, fn='yvh_all', key='yvh', loc='inpca_results_all/corners',
    #         ss=slice(0, None, 1), probs=True, ne=3, force=True,
    #         idx=idx, dev='cuda', distf='dbhat', reduction='mean',
    #         chunks=2000, proj=True, save_didx=True)
