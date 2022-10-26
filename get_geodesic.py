import torch as th
import h5py
import numpy as np
import pandas as pd

from utils import *
from reparameterization import *
from embed import lazy_embed


def main(loc="results/models/loaded", n=100):
    data = get_data()
    labels = {}
    qs = {}
    ps = {}
    for key in ["train", "val"]:
        k = "yh" if key == "train" else "yvh"
        y_ = np.array(data[key].targets, dtype=np.int32)
        y = np.zeros((y_.size, y_.max() + 1))
        y[np.arange(y_.size), y_] = 1
        qs[k] = np.sqrt(np.expand_dims(y, axis=0))
        ps[k] = np.sqrt(np.ones_like(qs[k]) / 10)
        labels[k] = y_

    ts = np.linspace(0, 1, n+1)[1:]
    geodesic = [
        dict(
            seed=0,
            bseed=-1,
            m="geodesic",
            opt="geodesic",
            t=1,
            err=0.0,
            favg=0.0,
            verr=0.0,
            vfavg=0.0,
            yh=ps["yh"],
            yvh=ps["yvh"],
        )
    ] * n
    geodesic = pd.DataFrame(geodesic)
    geodesic = geodesic.reindex(
        columns=[
            "seed",
            "bseed",
            "m",
            "opt",
            "t",
            "err",
            "favg",
            "verr",
            "vfavg",
            "bs",
            "drop",
            "aug",
            "bn",
            "lr",
            "wd",
            "yh",
            "yvh",
        ],
        fill_value="na",
    )

    for i in range(len(ts)):
        geodesic.loc[i, "t"] = ts[i]
        for key in ["yh", "yvh"]:
            geodesic.loc[i, key] = gamma(ts[i], ps[key], qs[key]) ** 2
            ekey = "err" if key == "yh" else "verr"
            fkey = "favg" if key == "yh" else "fvavg"
            geodesic.loc[i, ekey] = (
                1 - (np.argmax(geodesic.iloc[i][key], -1) == labels[key]).mean()
            )
            f = -np.take(geodesic.iloc[i][key], labels[key])[None, :, None]
            geodesic.loc[i, fkey] = f.mean()

    for key in ["yh", "yvh"]:
        geodesic[key] = geodesic.apply(lambda r: r[key].squeeze(), axis=1)

    d = dict(
        geodesic[
            ["seed", "bseed", "aug", "m", "bn", "drop", "opt", "bs", "lr", "wd"]
        ].iloc[0]
    )
    d["seed"] = 0
    d["bseed"] = -1
    fn = f"{json.dumps(d).replace(' ', '')}.p"
    print(os.path.join(loc, fn))
    th.save(geodesic, os.path.join(loc, fn))


def get_projection():
    key = "yh"
    idxs = ["seed", "m", "opt", "bs", "aug", "lr", "t", "wd"]
    didx_all = th.load(
        f"/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_{key}_all.p"
    )[idxs]
    for i in range(0, 2300, 100):
        didx_all = pd.concat(
            [didx_all, th.load(f"inpca_results/didx_{key}_geod_c{i}.p")["dc"][idxs]]
        )
    didx_all = didx_all.reset_index(drop=True)
    pairs = (
        didx_all[didx_all.duplicated(keep=False)]
        .groupby(list(didx_all))
        .apply(lambda x: tuple(x.index))
        .tolist()
    )
    ii = [i[0] for i in sorted(pairs, key=lambda x: x[1])]

    w_yh = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_geod.p")
    dp = w_yh[:, ii]
    th.save(dp, f"/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_geod.p")

    r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")
    d_mean = r["w_mean"]

    xp = lazy_embed(d_mean=d_mean, dp=dp, evals=r["e"], evecs=r["v"], ne=3)
    r["geod_extra"] = xp
    th.save(r, f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")


if __name__ == "__main__":
    main('results/models/reindexed_new', 100)
    # get_projection()