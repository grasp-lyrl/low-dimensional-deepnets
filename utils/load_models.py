import os
import json
import pandas as pd
import torch as th
import numpy as np
import math
from collections import defaultdict
import tqdm


def get_ts(bs, epochs, save_init=5, save_freq=4, len_data=50000):
    t = 0
    ts = [t]
    n_batches = len_data // bs
    for epoch in range(epochs):
        for i in range(n_batches):
            t += 1
            if epoch < save_init and i % (n_batches // save_freq) == 0:
                ts.append(t)

        if save_init <= epoch <= 25:
            ts.append(t)
        elif 25 < epoch <= 65 and epoch % 4 == 0:
            ts.append(t)
        elif epoch > 65 and epoch % 15 == 0 or (epoch == epochs - 1):
            ts.append(t)
    return ts


def get_idx(dd, cond):
    return dd.query(cond).index.tolist()

def c2fn(c):
    root = '/home/ubuntu/ext_vol/inpca/results/models/loaded/'
    if isinstance(c, list):
        return [c2fn(cc) for cc in c]
    else:
        fdict = dict(seed=int(c[0]), bseed=-1, aug=c[4], m=c[1], bn=True, drop=0., opt=c[2],
                    bs=int(c[3]), lr=float(c[-2]), wd=float(c[-1]), corner='normal', interp=False)
        fn = os.path.join(root, f'{json.dumps(fdict).replace(" ", "")}.p')
        return fn

def get_row_idx(dd, row, idxs=None):
    c = []
    if idxs is not None:
        dd = dd[idxs]
        row = row[idxs]
    for (k, v) in dict(row).items():
        if isinstance(v, str):
            v = f"'{v}'"
        c.append(f"{k} == {v}")
    c = " & ".join(c)
    return get_idx(dd, c)


def load_d(
    file_list,
    avg_err=False,
    numpy=True,
    probs=False,
    drop=0,
    keys=["yh", "yvh"],
    verbose=False,
    nmodels=-1,
    return_nan=False,
    loaded=False,
):
    r = [] 
    loaded_r = []
    nan_models = []
    count = 0
    nmodels = math.inf if nmodels < 0 else nmodels
    for f in tqdm.tqdm(file_list):
        configs = json.loads(f[f.find("{") : f.find("}") + 1])
        try:
            d_ = th.load(f)
        except RuntimeError:
            print(f)
            continue
        if loaded or isinstance(d_, pd.DataFrame):
            for c in configs:
                d_ = d_.assign(**{c: configs[c]})
            for k in keys:
                if (d_.iloc[0][k] < 0).any():
                    d_[k] = d_.apply(lambda r: np.exp(r[k]), axis=1)
                d_[k] = d_.apply(lambda r: r[k].squeeze(), axis=1)
            loaded_r.append(d_)
        else:
            d = d_["data"]
            dconf = vars(d_["configs"])
            if any(np.isnan(d[-1]["f"])):
                nan_models.append(configs)
                continue
            if dconf.get('data') == 'synthetic':
                ndata=dconf.get('num_train')
            else:
                ndata=50000

            ts =  get_ts(configs["bs"], dconf.get("epochs", 200), 
                         dconf.get('save_init', 5), dconf.get('save_freq', 4), len_data=ndata)
            for i in range(len(d)):
                t = {}
                t.update(configs)
                t.update({"t": ts[i]})
                t.update(d[i])
                for k in keys:
                    if not isinstance(t[k], np.ndarray):
                        t[k] = t[k].numpy()
                    if (t[k] < 0).any():
                        t[k] = np.exp(t[k])
                r.append(t)
            count += 1
            if count > nmodels:
                break
    if verbose:
        print(len(r), len(loaded_r))
    if len(r) > 0 or len(loaded_r) > 0:
        d = pd.DataFrame(r)
        if len(loaded_r) > 0:
            loaded_r = pd.concat(loaded_r)
            d = pd.concat([d, loaded_r])
        if avg_err and all(c in d.columns for c in ["e", "ev", "f", "fv"]):
            d["err"] = d.apply(lambda r: r.e.mean().item(), axis=1)
            d["verr"] = d.apply(lambda r: r.ev.mean().item(), axis=1)
            d["favg"] = d.apply(lambda r: r.f.mean().item(), axis=1)
            d["vfavg"] = d.apply(lambda r: r.fv.mean().item(), axis=1)

        if drop:
            d = drop_untrained(d, key="err", th=drop, verbose=True).reset_index()

        if verbose:
            print(d.keys(), len(d))
        del r

        if return_nan:
            return d, pd.DataFrame(nan_models)
        else:
            return d


def drop_untrained(dd, key="err", end_t=None, th=0.01, verbose=False):
    tmax = end_t or [dd["t"].max()]
    idxs = {t: get_idx(dd, f"t == {t} & {key} > {th}") for t in tmax}
    iis = []
    for (t, ii) in idxs.items():
        for i in ii:
            iis += list(range(i - t, i + 1, 1))
    if verbose:
        print(len(ii))
        print(dd[["m", "opt", "bn", "seed"]].iloc[ii])
    return dd.drop(iis)
