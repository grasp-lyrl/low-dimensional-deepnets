import torch as th
import h5py
import numpy as np
import pandas as pd

from utils import *
from reparameterization import *
from embed import lazy_embed


def main(loc="results/models/loaded", name='', n=100, ts=None, loaded=False, log=False,
         data_args={'data': 'CIFAR10', 'aug': 'none', 'sub_sample': 0}):
    data = get_data(data_args)
    labels = {}
    qs = {}
    ps = {}
    for key in ["train", "val"]:
        k = "yh" if key == "train" else "yvh"
        try:
            y_ = np.array(data[key].targets, dtype=np.int32)
        except AttributeError:
            y_ = np.array(data[key].y, dtype=np.int32)
        y = np.zeros((y_.size, y_.max() + 1))
        y[np.arange(y_.size), y_] = 1
        qs[k] = np.sqrt(np.expand_dims(y, axis=0))
        ps[k] = np.sqrt(np.ones_like(qs[k]) / 10)
        labels[k] = y_

    if ts is None:
        ts = np.linspace(0, 1, n+1)[1:]
    else:
        n = len(ts)
    geodesic = []

    for i in range(len(ts)):
        r = dict(
            seed=0,
            bseed=-1,
            m=f"{name}geodesic",
            opt=f"{name}geodesic",
        )
        r['t'] = ts[i]
        for key in ["yh", "yvh"]:
            r[key] = gamma(ts[i], ps[key], qs[key]) ** 2
            if log:
                r[key] = np.log(r[key])
            ekey = "e" if key == "yh" else "ev"
            fkey = "f" if key == "yh" else "fv"
            e = np.argmax(r[key], -1) == labels[key]
            r[ekey] = e.squeeze()
            errkey = "err" if key == "yh" else "verr"
            r[errkey] = 1 - e.mean()
            f = - (np.log(r[key]) * qs[key]).sum(-1)
            r[fkey] = f.squeeze()
            r[f'{fkey}avg'] = f.mean()
        geodesic.append(r)

    if loaded:
        geodesic = pd.DataFrame(geodesic)
        geodesic = geodesic.reindex(
            columns=[
                "seed",
                "bseed",
                "m",
                "opt",
                "t",
                "e",
                "ev",
                "err",
                "verr",
                "f",
                "fv",
                "favg",
                "fvavg",
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

        for key in ["yh", "yvh"]:
            geodesic[key] = geodesic.apply(lambda r: r[key].squeeze(), axis=1)

        d = dict(
            geodesic[
                ["seed", "bseed", "aug", "m", "bn", "drop", "opt", "bs", "lr", "wd"]
            ].iloc[0]
        )
    else:
        d = {k: geodesic[0][k] for k in ["seed", "bseed",
                                         "aug", "m", "bn", "drop", "opt", "bs", "lr", "wd"]}
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
    # ts = np.linspace(0, 1, 20000)
    # main(loc='results/models/moving_y', name='', ts=ts, loaded=True, log=False)
    data_args = get_configs('configs/data/synthetic-fc-50-0.5.yaml')
    ts = np.linspace(0.01, 1, 100)
    main(loc='results/models/sloppy-50', name='', ts=ts, loaded=True, log=False,
         data_args=data_args)
    # get_projection()
