import numpy as np
import scipy.linalg as sp
import torch as th

from itertools import product, combinations
from collections import defaultdict
import torch.multiprocessing as mp
from functools import partial
import os
import h5py
import json
import glob
import tqdm
import pandas as pd

from embed import xembed, proj_, lazy_embed
from utils import load_d, get_idx, CHOICES, dbhat


def get_idxs(s, file_list, idx=None):
    print(s)
    idx = idx or [
        "seed",
        "m",
        "opt",
        "t",
        "err",
        "favg",
        "verr",
        "vfavg",
        "bs",
        "aug",
        "bn",
        "lr",
        "wd",
    ]
    d = load_d(file_list=file_list[s], avg_err=True, probs=False, return_nan=False)
    didx = d[idx]
    th.save(didx, os.path.join("inpca_results", f"didx_{s}.p"))
    print(f"saved {s}")


def process_pair(pair, file_list, loc="inpca_results", keys=["yh"], save_didx=False):
    print(pair)
    s1, s2 = pair
    dist_from_flist(file_list[s1], file_list[s2], keys, loc, f"{s1}_{s2}", save_didx)


def dist_from_flist(f1, f2, keys=["yh", "yvh"], loc="", fn="", save_didx=False):
    d1 = load_d(
        file_list=f1,
        avg_err=False,
        probs=False,
        numpy=False,
        return_nan=False,
        loaded=True,
    )
    if f1 == f2:
        d2 = d1
    else:
        d2 = load_d(
            file_list=f2,
            avg_err=False,
            probs=False,
            numpy=False,
            return_nan=False,
            loaded=True,
        )
    for key in keys:
        idx = ["seed", "m", "opt", "t", "err", "verr", "bs", "aug", "lr", "wd"]
        xembed(
            d1,
            d2,
            fn=f"{key}_{fn}",
            ss=slice(0, -1, 2),
            probs=True,
            key=key,
            loc=loc,
            idx=idx,
            force=True,
            distf="dbhat",
            reduction="mean",
            chunks=3600,
            proj=False,
            save_didx=save_didx,
        )


def compute_distance(
    all_files=None,
    load_list=None,
    loc="results/models/reindexed",
    groupby=["seed", "m"],
    save_didx=False,
    save_loc="inpca_results_avg",
):
    if all_files is None:
        all_files = glob.glob(os.path.join(loc, "*}.p"))
    file_list = defaultdict(list)
    for f in all_files:
        configs = json.loads(f[f.find("{") : f.find("}") + 1])
        file_list[tuple(configs[key] for key in groupby)].append(f)

    if load_list is None:
        load_list_ = list(combinations(file_list.keys(), 2)) + [
            (k, k) for k in file_list.keys()
        ]
        load_list = []
        for pair in load_list_:
            s1, s2 = pair
            if not os.path.exists(os.path.join(save_loc, f"w_yh_{s1}_{s2}.p")):
                load_list.append(pair)

    for pair in load_list:
        if len(pair[0]) == 0 or len(pair[1]) == 0:
            continue
        process_pair(file_list=file_list, loc=save_loc, save_didx=save_didx, pair=pair)
    # mp.set_start_method('spawn')
    # with mp.Pool(processes=4) as pool:
    #     pool.map(
    #         partial(process_pair, file_list=file_list, loc=save_loc, save_didx=save_didx), load_list, chunksize=1)


def join():
    loc = "inpca_results"
    key = "yvh"
    seeds = range(42, 52)
    m = CHOICES["m"]
    rlist = None
    save_loc = "inpca_results_all"

    didxs_f = os.path.join(save_loc, f"didx_{key}_all.p")
    didxs = th.load(didxs_f)
    indices = didxs.groupby(["seed", "m"]).indices

    fname = os.path.join(save_loc, f"w_{key}_all.h5")
    if os.path.exists(fname):
        f = h5py.File(fname, "r+")
        dset = f["w"]
        dset.resize((len(didxs), len(didxs)))
    else:
        f = h5py.File(fname, "w")
        dset = f.create_dataset(
            "w", shape=(len(didxs), len(didxs)), maxshape=(None, None), chunks=True
        )

    groups = list(indices.keys())
    for i in range(len(groups)):
        r = groups[i]
        w_fname = os.path.join(loc, f"w_{key}_{r}_{r}.p")
        w_ = th.load(w_fname)
        ridxs = indices[r]

        def is_cts(l):
            return all((np.array(l[1:]) - np.array(l[:-1])) == 1)

        assert is_cts(ridxs)  # debug
        rstart, rend = ridxs[0], ridxs[-1] + 1
        dset[rstart:rend, rstart:rend] = w_

        cset = rlist or [groups[j] for j in range(i + 1, len(groups))]
        for c in cset:
            print(r, c)
            fname = os.path.join(loc, f"w_{key}_{r}_{c}.p")
            if os.path.exists(fname):
                w_ = th.load(fname)
            else:
                continue
            cidxs = indices[c]
            assert is_cts(cidxs)
            cstart, cend = cidxs[0], cidxs[-1] + 1
            dset[rstart:rend, cstart:cend] = w_
            dset[cstart:cend, rstart:rend] = w_.T
    f.close()


def project(seed=42, fn="yh_all", err_threshold=0.1, extra_points=None):
    loc = "inpca_results_all"
    ne = 3

    folder = os.path.join(loc, str(seed))
    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.isfile(os.path.join(folder, f"didx_{fn}.p")):
        idx, didx = th.load(os.path.join(folder, f"didx_{fn}.p"))
    else:
        # filter out un-trained model
        idx = []
        didx = th.load(os.path.join(loc, f"didx_{fn}.p"))
        for (c, indices) in didx.groupby(
            ["seed", "m", "opt", "bs", "aug", "lr", "wd"]
        ).indices.items():
            if (didx.iloc[indices[-1]]["err"] < err_threshold and c[0] == seed) or c[
                0
            ] == 0:
                idx.extend(indices)
        idx = sorted(idx)
        didx = didx.iloc[idx]
        th.save((idx, didx), os.path.join(folder, f"didx_{fn}.p"))

    f = h5py.File(os.path.join(loc, f"w_{fn}.h5"), "r")
    w = f["w"][:, idx][idx, :]
    n = w.shape[0]
    d_mean = w.mean(0)

    if os.path.isfile(os.path.join(folder, f"r_{fn}.p")):
        r = th.load(os.path.join(folder, f"r_{fn}.p"))
    else:
        l = np.eye(w.shape[0]) - 1.0 / w.shape[0]
        w = -l @ w @ l / 2
        r = proj_(w, n, ne)
        th.save(r, os.path.join(folder, f"r_{fn}.p"))

    if extra_points is not None:
        didx_ = th.load(os.path.join(loc, f"didx_{fn}.p"))
        ridx = get_idx(didx_, extra_points)
        dp = f["w"][:, idx][ridx, :]
        q = lazy_embed(dp=dp, d_mean=d_mean, evals=r["e"], evecs=r["v"], ne=ne)
        r["xp"] = np.vstack([r["xp"], q])
        didx = pd.concat([didx, didx_.iloc[ridx]])
        th.save((idx + ridx, didx), os.path.join(folder, f"didx_{fn}.p"))

    th.save(r, os.path.join(folder, f"r_{fn}_all.p"))


def compute_path_distance(
    loc="results/models/reindexed",
    all_files=None,
    load=False,
    save_loc="inpca_results_avg",
    key="yh",
    fn="pointwise",
    T=100,
):
    if all_files is None:
        all_files = glob.glob(os.path.join(loc, "*}.p"))
        all_files.sort(key=os.path.getmtime)
        all_files = np.array(all_files)
    idx = ["seed", "m", "opt", "err", "verr", "bs", "aug", "lr", "wd"]
    if load:
        didx = th.load(os.path.join(save_loc, f"didx_{fn}_{key}.p")).reset_index(
            drop=True
        )
        dists = th.load(os.path.join(save_loc, f"dists_{fn}_{key}.p"))
        chunks = np.array_split(np.arange(len(didx)), 10)
        chunks.extend(np.array_split(np.arange(len(didx), len(all_files)), 2))
    else:
        didx = None
        dists = np.zeros([T, len(all_files), len(all_files)])
        chunks = np.array_split(np.arange(len(all_files)), 10)
    print(len(all_files))

    with tqdm.tqdm(total=len(chunks) ** 2 // 2) as pbar:
        for i in range(len(chunks)):
            i1 = chunks[i]

            d1 = load_d(
                file_list=all_files[i1],
                avg_err=False,
                verbose=False,
                numpy=False,
                return_nan=False,
                loaded=True,
            ).reset_index(drop=True)
            d1 = d1.rename(columns={"train_err": "err", "val_err": "verr"})
            d1['t'] = d1['t'].round(2)
            t1 = d1.groupby(["t"]).indices
            assert T == len(t1)
            if np.allclose(dists[:, i1[0] : i1[-1] + 1, i1[0] : i1[-1] + 1], 0):
                didx = pd.concat([didx, d1.iloc[t1[1.0]][idx]], ignore_index=True)
                th.save(didx, os.path.join(save_loc, f"didx_{fn}_{key}.p"))

            for j in tqdm.tqdm(range(i, len(chunks))):
                i2 = chunks[j]
                if not np.allclose(dists[:, i1[0] : i1[-1] + 1, i2[0] : i2[-1] + 1], 0):
                    print(i, j)
                    continue
                d2 = load_d(
                    file_list=all_files[i2],
                    avg_err=False,
                    verbose=False,
                    numpy=False,
                    return_nan=False,
                    loaded=True,
                ).reset_index(drop=True)
                d2 = d2.rename(columns={"train_err": "err", "val_err": "verr"})
                d2['t'] = d2['t'].round(2)
                t2 = d2.groupby(["t"]).indices
                for (ti, (t, ii)) in enumerate(t1.items()):
                    x1 = np.stack(d1.iloc[ii][key])
                    x2 = np.stack(d2.iloc[t2[t]][key])
                    if not (len(x1) == len(i1) and len(x2) == len(i2)):
                        import ipdb; ipdb.set_trace()
                    assert np.allclose(
                        dists[ti, i1[0] : i1[-1] + 1, i2[0] : i2[-1] + 1], 0
                    )
                    dists[ti, i1[0] : i1[-1] + 1, i2[0] : i2[-1] + 1] = dbhat(
                        th.Tensor(x1), th.Tensor(x2), reduction="mean", chunks=50
                    )
                if j > i:
                    assert np.allclose(
                        dists[:, i2[0] : i2[-1] + 1, i1[0] : i1[-1] + 1], 0
                    )
                    dists[:, i2[0] : i2[-1] + 1, i1[0] : i1[-1] + 1] = dists[
                        :, i1[0] : i1[-1] + 1, i2[0] : i2[-1] + 1
                    ].transpose(0, 2, 1)
                pbar.update(1)
                th.save(
                    dists,
                    os.path.join(save_loc, f"dists_{fn}_{key}.p"),
                    pickle_protocol=4,
                )


if __name__ == "__main__":
    # compute_distance(loc='results/models/reindexed_new', groupby=['seed', 'm'], save_loc='inpca_results_avg_new')

    ################################
    # compute distance to geodesic #
    ################################
    # file_list = glob.glob('results/models/loaded/*}.p')
    # geod = 'results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na"}.p'

    # for i in range(0, len(file_list), 100):
    #     dist_from_flist([geod], file_list[i:i+100], loc='inpca_results', fn=f'geod_c{i}', save_didx=True)

    ##############################
    # join the distance matrices #
    ##############################
    cols = ['seed', 'm', 'opt', 't', 'bs', 'aug',  'lr', 'wd']
    dall = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all.p').reset_index(drop=True)
    dall = dall.reset_index(drop=False)

    d = th.load('/home/ubuntu/ext_vol/inpca/inpca_results/didx_yh_geod_c0.p')['dc']
    w = th.load('/home/ubuntu/ext_vol/inpca/inpca_results/w_yh_geod_c0.p')
    for i in range(100, 2300, 100):
        d = pd.concat([d, th.load(f'inpca_results/didx_yh_geod_c{i}.p')['dc']])
        w = np.hstack([w, th.load(f'inpca_results/w_yh_geod_c{i}.p')])
    mask = d[cols].duplicated(keep='first')
    d = d[~mask]
    w = w[:, ~mask]
    d = d.reset_index(drop=True)
    d = d.reset_index(drop=False)
    
    dmerged = pd.merge(dall, d, on=cols, how='inner')
    ii = dmerged['index_y'].values

    f = h5py.File('/home/ubuntu/ext_vol/inpca/inpca_results_all/w_yh_all_geod.h5', 'r+')
    wall = f['w']
    wall[-100:, :] = w[:, ii]
    wall[:, -100:] = w[:, ii].T
    f.close()

    ################################################
    # compute pairwise distance for particle & avg #
    ################################################
    # file_list = glob.glob('/home/ubuntu/ext_vol/inpca/results/models/reindexed_new/*}.p')
    # all_files = []
    # for f in file_list:
    #     c = json.loads(f[f.find('{'):f.find('}')+1])
    #     if c['seed'] <= 0:
    #         all_files.append(f)
    # ms = ['wr-10-4-8', 'convmixer', 'allcnn', 'wr-16-4-64']
    # load_list = [((0, 'geodesic'), (0, 'geodesic'))]
    # print(load_list)
    # compute_distance(all_files=all_files, load_list=load_list,
    # loc='results/models/reindexed_new', groupby=['seed', 'm'], save_loc='inpca_results_avg_new', save_didx=True)

    ##################################################
    # compute 3d distance tensor, including geodesic #
    ##################################################
    all_files = glob.glob(os.path.join("results/models/reindexed_new", "*}.p"))
    all_files = np.array(all_files)

    compute_path_distance(
        all_files=all_files,
        loc="results/models/reindexed_new",
        save_loc="inpca_results_avg_new",
        load=True,
        fn="3d_geod",
    )
