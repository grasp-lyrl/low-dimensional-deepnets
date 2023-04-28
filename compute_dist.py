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

from utils.embed import xembed, proj_, lazy_embed
from utils import load_d, get_idx, CHOICES, dbhat


def join_didx(loc="inpca_results", key="yh", fn="", 
              load_list=[],
              groupby=["m"], remove_f=False):
    if len(load_list) == 0:
        load_list = [(c, c) for c in product(*[CHOICES[g] for g in groupby])]
    loaded = set()
    d = pd.DataFrame()
    for pair in load_list:
        c1, c2 = pair
        load_fn = os.path.join(loc, f"didx_{key}_{c1}_{c2}.p")
        if os.path.exists(load_fn):
            if not c1 in loaded:
                loaded.add(c1)
                di = th.load(load_fn)["dr"]
                d = pd.concat([d, di])
            if not c2 in loaded:
                loaded.add(c2)
                di = th.load(load_fn)["dc"]
                d = pd.concat([d, di])
            if remove_f:
                os.remove(load_fn)
    save_fn = os.path.join(loc, f"didx_{fn}.p")
    th.save(d.reset_index(drop=True), save_fn)
    print("saved ", save_fn)


def dist_from_flist(f1, f2=None, keys=["yh", "yvh"], loc="", fn="",
                    ss=slice(0, -1, 2),
                    dev='cuda',
                    distf="dbhat",
                    idx=["seed", "m", "opt", "t", "err", "verr", "bs", "aug", "lr", "wd"],
                    avg_err=True, loaded=False, save_didx=False):
    d1 = load_d(
        file_list=f1,
        avg_err=avg_err,
        numpy=loaded,
        loaded=loaded,
    )
    if f2 is not None:
        d2 = load_d(
            file_list=f2,
            avg_err=avg_err,
            numpy=loaded,
            loaded=loaded,
        )
    else:
        d2 = None
    for key in keys:
        xembed(
            d1,
            d2,
            fn=f"{key}_{fn}",
            ss=ss,
            probs=True,
            key=key,
            loc=loc,
            idx=idx,
            dev=dev,
            force=True,
            distf=distf,
            reduction="mean",
            chunks=1200,
            proj=False,
            save_didx=save_didx,
        )


def process_pair(pair, file_list, loc="inpca_results",
                 idx=["seed", "m", "opt", "t", "err",
                      "verr", "bs", "aug", "lr", "wd"],
                 distf="dbhat",
                 save_didx=False,
                 keys=["yh", "yvh"]):
    print(pair)
    s1, s2 = pair
    save_didx = (s1 == s2) or save_didx
    dist_from_flist(f1=file_list[s1], f2=file_list[s2], keys=keys, 
                    idx=idx, distf=distf,
                    loc=loc, fn=f"{s1}_{s2}", save_didx=save_didx)


def compute_distance(
    all_files=None,
    load_list=None,
    loc="results/models/reindexed",
    groupby=["seed", "m"],
    save_didx=False,
    distf="dbhat",
    save_loc="inpca_results_avg",
    idx=["seed", "m", "opt", "t", "err", "verr", "bs", "aug", "lr", "wd"],
    parallel=-1,
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
            if (not os.path.exists(os.path.join(save_loc, f"w_yh_{s1}_{s2}.p"))) \
                or (not os.path.exists(os.path.join(save_loc, f"w_yvh_{s1}_{s2}.p"))):
                load_list.append(pair)

    print(len(load_list))
    if parallel > 0:
        with mp.Pool(processes=parallel) as pool:
            pool.map(
                partial(process_pair, file_list=file_list, save_didx=save_didx, 
                        distf=distf,
                        loc=save_loc, idx=idx), load_list, chunksize=1)
    else:
        for pair in load_list:
            if len(pair[0]) == 0 or len(pair[1]) == 0:
                continue
            process_pair(file_list=file_list, loc=save_loc, 
                        save_didx=save_didx,
                        distf=distf,
                        idx=idx,
                        pair=pair)
    return load_list


def merge_dists(fn1, fn2, merge_loc="/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/", 
                key="yh", save_f="merged_allcnn",
                cols= ['seed', 'iseed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd', 'corner', 'isinit', 't']):
    # merge two distance matrices, keeping the order of the first didx
    d1 = th.load(os.path.join(merge_loc, f"didx_{fn1}.p")).reset_index()
    d2 = th.load(os.path.join(merge_loc, f"didx_{fn2}.p")).reset_index()
    didxs = d1.merge(d2, on=cols, how='outer')
    th.save(didxs, os.path.join(merge_loc, f"didx_{save_f}.p"))

    id1 = np.array(didxs.index_y.dropna().index)
    id2 = np.array(didxs.index_y.dropna().values.astype(int))

    with h5py.File(os.path.join(merge_loc, f'w_{key}_{fn2}.h5'), 'r') as f:
        w = f['w'][:]
    wi = w[id2, :][:, id2]
    
    save_fn = os.path.join(merge_loc, f'w_{key}_{save_f}.h5')
    if os.path.exists(save_fn):
        f = h5py.File(save_fn, "r+")
        dset = f["w"]
        dset.resize((len(didxs), len(didxs)))
    else:
        f = h5py.File(save_fn, "w")
        dset = f.create_dataset("w", shape=(len(didxs), len(didxs)), maxshape=(None, None), chunks=True)
        with h5py.File(os.path.join(merge_loc, f'w_{key}_{fn1}.h5'), 'r') as f:
            w = f['w'][:]
        dset[:, :] = w

    def is_cts(l):
        return all((np.array(l[1:]) - np.array(l[:-1])) == 1)
    assert is_cts(id1) 
    dset[id1[0]:id1[-1]+1, id1[0]:id1[-1]+1] = wi
    f.close()


def join(loc="inpca_results_avg_new", key="yh", groupby=["m"], 
         save_loc="inpca_results_avg_new", fn="all", remove_f=False):

    didxs_f = os.path.join(save_loc, f"didx_{fn}.p")
    didxs = th.load(didxs_f)
    didxs = didxs.reset_index(drop=True)

    indices = didxs.groupby(groupby).indices

    fname = os.path.join(save_loc, f"w_{key}_{fn}.h5")
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
    pairs = list(combinations(groups, 2)) + [(r, r) for r in groups]
    for pair in tqdm.tqdm(pairs):
        r, c = pair
        ridxs, cidxs = indices[r], indices[c]
        r = (r, ) if not isinstance(r, tuple) else r
        c = (c, ) if not isinstance(c, tuple) else c
        fname = os.path.join(loc, f"w_{key}_{r}_{c}.p")
        if not os.path.exists(fname):
            fname = os.path.join(loc, f"w_{key}_{c}_{r}.p")
            c, r = pair
            cidxs, ridxs = indices[c], indices[r]
        try:
            w_ = th.load(fname)
        except:
            print(fname)
            continue

        def is_cts(l):
            return all((np.array(l[1:]) - np.array(l[:-1])) == 1)

        assert is_cts(ridxs) and is_cts(cidxs)  # debug
        rstart, rend = int(ridxs[0]), int(ridxs[-1]) + 1
        cstart, cend = int(cidxs[0]), int(cidxs[-1]) + 1
        dset[rstart:rend, cstart:cend] = w_
        dset[cstart:cend, rstart:rend] = w_.T
        if remove_f:
            os.remove(fname)
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
    idx = ["seed", "m", "opt", "err", "verr", "bs", "aug", "lr", "wd"],
    T=100,
):
    if all_files is None:
        all_files = glob.glob(os.path.join(loc, "*}.p"))
        all_files = np.array(all_files)
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
        chunks = np.array_split(np.arange(len(all_files)), 30)
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
            )
            if d1 is None:
                import ipdb; ipdb.set_trace()
            d1 = d1.reset_index(drop=True)
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
                        th.Tensor(x1), th.Tensor(x2), reduction="mean", chunks=50, 
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


    ##################################################
    # compute 3d distance tensor, including geodesic #
    ##################################################

    # compute_path_distance(
    #     all_files=all_files[58:],
    #     loc="results/models/reindexed_all",
    #     save_loc="inpca_results_avg_new",
    #     load=False,
    #     idx = ["seed", "m", "opt", "err_interp", "verr_interp", "bs", "aug", "lr", "wd"],
    #     key="yh",
    #     fn="3d_geod",
    # )

    # def fn_from_config(c, root='/home/ubuntu/ext_vol/inpca/results/models/all/'):
    #     if c[0] == 0:
    #         return '/home/ubuntu/ext_vol/inpca/results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na","interp":false}.p'
    #     else:
    #         if len(c) == 8:
    #             corner=c[-1]
    #         else:
    #             corner = 'normal'
    #         dic = {'seed':int(c[0]), 'bseed':-1, 'aug':c[1], 'm':c[2], 
    #                     'bn':True, 'drop':0.0, 'opt':c[4], 'bs':int(c[5]), 'lr':float(c[3]), 'wd':float(c[6]),
    #                     'corner':corner}
    #         fn = json.dumps(dic).replace(' ', '') + '.p'
    #         return os.path.join(root, fn)

    # didx = pd.DataFrame(th.load(
    #     '/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/didx_yh_all.p')['dr'])

    # cols = ['aug', 'm', 'lr', 'opt', 'bs', 'wd', 'corner']
    # keys = didx.groupby(cols).indices.keys()
    # ll = [tuple([i] + list(r[:-1]) + [c]) for i in range(42, 52) for r in keys for c in ['normal', r[-1]]]
    # fs = [fn_from_config(c) for c in ll]
    # print(len(fs))

    ################################################
    # compute pairwise distance for synthetic data #
    ################################################
    # file_list = {}
    # for c in [0.5]:
    #     data = f'sloppy-50-{c}'
    #     all_files = glob.glob(
    #         f'/home/ubuntu/ext_vol/inpca/results/models/{data}/*.p')
    #     dfs = pd.DataFrame(
    #         [json.loads(f[f.find("{"): f.find("}") + 1]) for f in all_files])
    #     di = list(dfs[(dfs.init != 'perturbed_normal')
    #             & (dfs.m != 'geodesic') 
    #             & (dfs.init != 'fixed_var_normal')].index)
    #     file_list[f'sloppy_{c}_ignorance'] = np.array(all_files)[di]

    # mp.set_start_method('spawn')
    # from eigvals import main
    # group=["m"]

    # for (fn, fs) in file_list.items():
    #     compute_distance(
    #         all_files=fs,
    #         groupby=group,
    #         idx=["seed", "m", "opt", "t", "err",
    #             "verr", "bs", "aug", "lr", "wd"],
    #         save_loc="inpca_results_all/synth",
    #         parallel=1
    #     )

    #     for k in ['yh', 'yvh']:
    #         join_didx(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, remove_f=True)
    #         join(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, save_loc="inpca_results_all/synth", remove_f=True)
    #         main(fn=f'{k}_{fn}', save_fn=f'{k}_{fn}', cond='', root="/home/ubuntu/ext_vol/inpca/inpca_results_all/synth")

    ##########################
    # synthetic data corners #
    ##########################
    # fs = {}
    # for ncorners in [1, 2, 5, 10]:
    #     iseeds = np.arange(ncorners)
    #     seeds = np.arange(42, 92)[:int(50/ncorners)]
    #     fs[ncorners] = np.array(all_files)[list(
    #         dfs[(dfs.seed.isin(seeds)) & (dfs.iseed.isin(iseeds)) & (dfs.init=='perturbed_normal')].index)]

    # mp.set_start_method('spawn')
    # from eigvals import main
    # for (i, f) in fs.items():
    #     print(len(f))
    #     fn = f"{i}_{data}"
    #     group = ["opt", "bs"]
    #     compute_distance(
    #         all_files=f,
    #         groupby=group,
    #         idx=["seed", "iseed", "m", "opt", "t", "err", "init",
    #             "verr", "bs", "aug", "lr", "wd"],
    #         save_loc="inpca_results_all/synth",
    #         parallel=2
    #     )

    #     for k in ['yh', 'yvh']:
    #         join_didx(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, remove_f=True)
    #         join(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, save_loc="inpca_results_all/synth", remove_f=True)
    #         main(fn=f'{k}_{fn}', save_fn=f'{k}_{fn}', cond='', root="/home/ubuntu/ext_vol/inpca/inpca_results_all/synth")

    #########################################
    # compute pairwise distance for corners #
    #########################################
    fs_all = glob.glob('/home/ubuntu/ext_vol/inpca/results/models/corners/*.p')
    fs_normal_all = glob.glob(
        '/home/ubuntu/ext_vol/inpca/results/models/loaded/*.p')
    fs_normal = []
    for f in fs_normal_all:
        configs = json.loads(f[f.find("{"): f.find("}") + 1])
        if configs['m'] == 'allcnn' or configs['m'] == 'geodesic':
            fs_normal.append(f)

    fs = fs_all + fs_normal
    fn = 'allcnn_all_geod'
    group = ["seed"]
    load_list = [((0,), (s,)) for s in [0] + list(range(42, 52))]

    from eigvals import main
    mp.set_start_method('spawn')
    load_list = compute_distance(
        all_files=fs,
        groupby=group,
        load_list=load_list,
        idx=["seed", "iseed", "m", "opt", "t", "err", "isinit",
             "verr", "bs", "aug", "lr", "wd", "corner"],
        save_loc="inpca_results_all/corners",
        save_didx=True,
        parallel=4
    )

    for k in ['yh', 'yvh']:
        join_didx(loc="inpca_results_all/corners", key=k, fn=fn,
                  groupby=group, remove_f=True, load_list=load_list)
        loc = "/home/ubuntu/ext_vol/inpca/inpca_results_all/corners"
        join(loc=loc, key=k, fn="allcnn_all_geod", groupby=["seed"],
            save_loc=loc, remove_f=True)
        merge_dists("allcnn_all_geod", "allcnn_geod", merge_loc=loc, 
                key=k, save_f="merged_allcnn",
                cols= ['seed', 'iseed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd', 'corner', 'isinit', 't'])
        main(key=k, load_fn=f'merged_allcnn', save_fn=f'merged_allcnn', cond='',
             root=loc)


    #######################################
    # compute pairwise Euclidean Distance #
    # #####################################
    # all_f = glob.glob('/home/ubuntu/ext_vol/inpca/results/models/loaded/*.p')
    # fs = []
    # for f in all_f:
    #     configs = json.loads(f[f.find("{"): f.find("}") + 1])
    #     if configs['m'] == 'extended_geodesic' or configs['seed'] == 52:
    #         continue
    #     fs.append(f)
    # group=["seed", "opt"]
    
    # mp.set_start_method('spawn')
    # from eigvals import main
    # load_list = compute_distance(
    #     all_files=fs,
    #     groupby=group,
    #     idx=["seed", "m", "opt", "t", "err", "favg", "vfavg",
    #          "verr", "bs", "aug", "lr", "wd"],
    #     save_loc="inpca_results_all/euclidean",
    #     distf="deuclid",
    #     save_didx=True,
    #     parallel=2
    # )

    # fn = "all_euclid"
    # for k in ['yh', 'yvh']:
    #     join_didx(loc="inpca_results_all/euclidean", key=k, fn=fn,
    #               load_list=load_list,
    #               groupby=group, remove_f=False)
    #     join(loc="inpca_results_all/euclidean", key=k, fn=fn, groupby=group,
    #          save_loc="inpca_results_all/euclidean", remove_f=False)
    #     main(fn=f'{k}_{fn}', save_fn=f'{k}_{fn}', cond='',
    #          root="/home/ubuntu/ext_vol/inpca/inpca_results_all/euclidean")

    pass
