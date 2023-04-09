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


def join_didx(loc="inpca_results", key="yh", fn="", groupby=["m"], remove_f=False):
    load_list_ = product(*[CHOICES[g] for g in groupby])
    d = pd.DataFrame()
    for c in load_list_:
        load_fn = os.path.join(loc, f"didx_{key}_{c}_{c}.p")
        if os.path.exists(load_fn):
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
            distf="dbhat",
            reduction="mean",
            chunks=800,
            proj=False,
            save_didx=save_didx,
        )


def process_pair(pair, file_list, loc="inpca_results",
                 idx=["seed", "m", "opt", "t", "err",
                      "verr", "bs", "aug", "lr", "wd"],
                  keys=["yh", "yvh"]):
    print(pair)
    s1, s2 = pair
    save_didx = (s1 == s2)
    dist_from_flist(f1=file_list[s1], f2=file_list[s2], keys=keys, 
                    idx=idx, 
                    loc=loc, fn=f"{s1}_{s2}", save_didx=save_didx)


def compute_distance(
    all_files=None,
    load_list=None,
    loc="results/models/reindexed",
    groupby=["seed", "m"],
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
                partial(process_pair, file_list=file_list, loc=save_loc, idx=idx), load_list, chunksize=1)
    else:
        for pair in load_list:
            if len(pair[0]) == 0 or len(pair[1]) == 0:
                continue
            process_pair(file_list=file_list, loc=save_loc, 
                        idx=idx,
                        pair=pair)

def join(loc="inpca_results_avg_new", key="yh", groupby=["m"], 
         save_loc="inpca_results_avg_new", fn="all", remove_f=False):

    didxs_f = os.path.join(save_loc, f"didx_{fn}.p")
    didxs = th.load(didxs_f).reset_index(drop=True)
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
        rstart, rend = ridxs[0], ridxs[-1] + 1
        cstart, cend = cidxs[0], cidxs[-1] + 1
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

    ################################
    # compute distance to geodesic #
    ################################
    # geod = '/home/ubuntu/ext_vol/inpca/results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na"}.p'

    # dist_from_flist([geod], [geod], loc='inpca_results', fn=f'{fn}geod_geod', save_didx=True)
    # for i in range(0, len(flist), 100):
    #     dist_from_flist([geod], flist[i:i+100], loc='inpca_results', fn=f'{fn}geod_c{i}', save_didx=True)

    ##############################
    # join the distance matrices #
    ##############################
    # fn = ''
    # key = "yh"
    # cols = ['seed', 'm', 'opt', 't', 'bs', 'aug',  'lr', 'wd']
    # dall = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_{key}_all.p').reset_index(drop=True)
    # dgeod = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results/didx_{key}_{fn}geod_c0.p')['dr']
    # dall = pd.concat([dall, dgeod], ignore_index=True)
    # dall = dall.reset_index(drop=False)
    # th.save(dall, f'/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all.p')

    # d = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results/didx_{key}_{fn}geod_c0.p')['dc']
    # w = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results/w_{key}_{fn}geod_c0.p')
    # for i in range(100, 2300, 100):
    #     di = th.load(f'inpca_results/didx_{key}_{fn}geod_c{i}.p')['dc'].reset_index(drop=True)
    #     wi = th.load(f'inpca_results/w_{key}_{fn}geod_c{i}.p')
    #     if len(di) != w.shape[1]:
    #         wi = wi[:, di.index]
    #     d = pd.concat([d, di])
    #     w = np.hstack([w, wi])

    # mask = d[cols].duplicated(keep='first')
    # d = d[~mask]
    # w = w[:, ~mask]
    # d = d.reset_index(drop=True)
    # d = d.reset_index(drop=False)
    
    # d['t'] = d['t'].astype(float)
    # d[cols] = d[cols].astype(str)
    # dall[cols] = dall[cols].astype(str)
    # dmerged = pd.merge(dall, d, on=cols, how='inner')
    # ii = dmerged['index_y'].values

    # f = h5py.File(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_all_geod.h5', 'r+')
    # wall = f['w']
    # if wall.shape[0] != len(dall):
    #     f['w'].resize([len(dall), len(dall)])
    # wall[-len(dgeod):, :] = w[:, ii]
    # wall[:, -len(dgeod):] = w[:, ii].T
    # # wall[-len(dgeod):, -len(dgeod):] = th.load(f'inpca_results/w_{key}_{fn}geod_geod.p')
    # f.close()

    ################################################
    # compute pairwise distance for particle & avg #
    ################################################
    # ms = ['wr-10-4-8', 'convmixer', 'allcnn', 'wr-16-4-64']
    # load_list = [((0, 'geodesic'), (0, 'geodesic'))]
    # print(load_list)
    # compute_distance(all_files=all_files, load_list=load_list,
    # loc='results/models/reindexed_new', groupby=['seed', 'm'], save_loc='inpca_results_avg_new', save_didx=True)

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

    #########################################
    # compute pairwise distance for corners #
    #########################################
    # fs_all = glob.glob('/home/ubuntu/ext_vol/inpca/results/models/corners/*.p')
    # fs_noinit = []

    # for f in fs_all:
    #     configs = json.loads(f[f.find("{"): f.find("}") + 1])
    #     if not configs['isinit']:
    #         fs_noinit.append(f)

    # fs_normal_all = glob.glob('/home/ubuntu/ext_vol/inpca/results/models/loaded/*.p')
    # fs_normal = []
    # for f in fs_normal_all:
    #     configs = json.loads(f[f.find("{"): f.find("}") + 1])
    #     if configs['m'] == 'allcnn':
    #         fs_normal.append(f)

    # fs = fs_all + fs_normal + \
    #     ['/home/ubuntu/ext_vol/inpca/results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na","interp":false}.p']
    # fn = 'allcnn_geod_debug'

    data = 'isog-50'
    all_files = glob.glob(
        f'/home/ubuntu/ext_vol/inpca/results/models/{data}/*.p')
    dfs = pd.DataFrame(
        [json.loads(f[f.find("{"): f.find("}") + 1]) for f in all_files])
    fs = {}
    for ncorners in [1, 2, 5, 10]:
        iseeds = np.arange(ncorners)
        seeds = np.arange(42, 92)[:int(50/ncorners)]
        fs[ncorners] = np.array(all_files)[list(
            dfs[(dfs.seed.isin(seeds)) & (dfs.iseed.isin(iseeds)) & (dfs.init=='perturbed_normal')].index)]

    mp.set_start_method('spawn')
    from eigvals import main
    for (i, f) in fs.items():
        print(len(f))
        fn = f"{i}_{data}"
        group = ["opt", "bs"]
        compute_distance(
            all_files=f,
            groupby=group,
            idx=["seed", "iseed", "m", "opt", "t", "err", "init",
                "verr", "bs", "aug", "lr", "wd"],
            save_loc="inpca_results_all/synth",
            parallel=2
        )

        for k in ['yh', 'yvh']:
            join_didx(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, remove_f=True)
            join(loc="inpca_results_all/synth", key=k, fn=fn, groupby=group, save_loc="inpca_results_all/synth", remove_f=True)
            main(fn=f'{k}_{fn}', save_fn=f'{k}_{fn}', cond='', root="/home/ubuntu/ext_vol/inpca/inpca_results_all/synth")