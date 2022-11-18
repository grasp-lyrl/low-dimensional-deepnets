import torch as th
import h5py
import numpy as np
import pandas as pd

from utils import *
from reparameterization import *
from embed import lazy_embed


def main(loc="results/models/loaded", name='', n=100, ts=None):
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

    if ts is None:
        ts = np.linspace(0, 1, n+1)[1:]
    else:
        n = len(ts)
    geodesic = [
        dict(
            seed=0,
            bseed=-1,
            m=f"{name}geodesic",
            opt=f"{name}geodesic",
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
    f = h5py.File(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_all_geod.h5", 'r')
    dp = f['w'][-100:, :-100]
    f.close()

    r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")
    d_mean = r["w_mean"]

    xp = lazy_embed(d_mean=d_mean, dp=dp, evals=r["e"], evecs=r["v"], ne=3)
    r["extra_points"] = xp
    th.save(r, f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all.p")


if __name__ == "__main__":
    t1 = np.linspace(0, 0.9, 100)
    t2 = np.linspace(0.9, 1, 100)[1:]
    ts = np.hstack([t1, t2])
    main(loc='results/models/loaded', name='extended_', ts=ts)
    # get_projection()