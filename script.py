import h5py
from utils.embed import *
from compute_dist import *

############################################
# embed interp models with original models #
############################################
# dall = th.load(
#     "/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_yh_all.p").reset_index(drop=True)
# dinterp = th.load(
#     "/home/ubuntu/ext_vol/inpca/inpca_results_avg_new/didx_all.p").reset_index(drop=True)
# idxs_all = get_idx(dall, "aug=='none' & seed <=46 & opt=='sgd' & bs <= 500")
# dall = dall.iloc[idxs_all].reset_index(drop=True)
# idxs_interp = get_idx(dinterp, "aug=='none' & seed <=46 & opt=='sgd' & bs <= 500")
# dinterp = dinterp.iloc[idxs_interp].reset_index(drop=True)

# w1 = h5py.File(
#     '/home/ubuntu/ext_vol/inpca/inpca_results_all/w_yh_all.h5', 'r')['w']
# w1 = w1[idxs_all, :][:, idxs_all]
# w2 = h5py.File(
#     '/home/ubuntu/ext_vol/inpca/inpca_results_avg_new/w_yh_all.h5', 'r')['w']
# w2 = w2[idxs_interp, :][:, idxs_interp]

# didx_cross = th.load(
#     '/home/ubuntu/ext_vol/inpca/inpca_results_mixed/didx_yh_subset.p')
# w3 = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_mixed/w_yh_subset.p')

# n1, n2= w3.shape
# w = np.zeros([n1+n2, n1+n2])
# w[:n1, :n1] = w1
# w[-n2:, -n2:] = w2
# w[:n1, -n2:] = w3
# w[-n2:, :n1] = w3.T
# l = np.eye(w.shape[0]) - 1.0/w.shape[0]
# w = -l @ w @ l / 2
# r = proj_(w, w.shape[0], 3)

# th.save(r, '/home/ubuntu/ext_vol/inpca/inpca_results_mixed/r_yh_subset.p')

# dall = th.load(
#     "/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_yh_all.p").reset_index(drop=True)
# dinterp = th.load(
#     "/home/ubuntu/ext_vol/inpca/inpca_results_avg_new/didx_all.p").reset_index(drop=True)
# idxs = get_idx(dall, "aug=='none' & seed <=46 & opt=='sgd' & bs <= 500")
# dall = dall.iloc[idxs].reset_index(drop=True)
# idxs = get_idx(dinterp, "aug=='none' & seed <=46 & opt=='sgd' & bs <= 500")
# dinterp = dinterp.iloc[idxs].reset_index(drop=True)
# cols = ['seed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd']
# dall_idxs = dall.groupby(cols).indices
# dinterp_idxs = dinterp.groupby(cols).indices
# interpf = []
# allf = []
# all_root = '/home/ubuntu/ext_vol/inpca/results/models/loaded'
# interp_root = '/home/ubuntu/ext_vol/inpca/results/models/reindexed_new'
# for c in dall_idxs.keys():
#     seed, m, opt, bs, aug, lr, wd = c

#     fn = json.dumps(dict(seed=int(seed), bseed=-1, aug=aug, m=m, bn=True, drop=0.0,
#                             opt=opt, bs=int(bs), lr=lr, wd=wd, corner='normal', interp=False)).replace(" ", "")
#     allf.append(os.path.join(all_root, f'{fn}.p'))
#     fn = json.dumps(dict(seed=int(seed), bseed=-1, aug=aug, m=m, bn=True, drop=0.0,
#                             opt=opt, bs=int(bs), lr=lr, wd=wd, interp=True)).replace(" ", "")
#     interpf.append(os.path.join(interp_root, f'{fn}.p'))
# dist_from_flist(f1=allf, f2=interpf, keys=["yh"],
#                 loc="inpca_results_mixed", fn=f"subset", save_didx=True)

#######################
# compute error count #
#######################
# all_files = []
# for f in glob.glob(os.path.join("results/models/loaded", "*}.p")):
#     configs = json.loads(f[f.find("{"): f.find("}") + 1])
#     if 42 <= configs['seed'] <= 46 and configs['aug'] == 'none':
#         all_files.append(f)

# cols = ['seed', 'aug', 'm', 'opt', 'bs', 'lr', 'wd',
#         'err', 'verr', 'favg', 'vfavg', 'lam_yh', 'lam_yvh']
# dd = []
# ee = []
# ev = []
# for f in tqdm.tqdm(all_files):
#     try:
#         d = th.load(f)
#     except RuntimeError:
#         continue
#     # dd.append(dict(d.iloc[-1][cols]))
#     # ee.append(d.iloc[-1].e)
#     ev.append(d.iloc[-1].ev)

# # dd = pd.DataFrame(dd)
# # ee = np.stack(ee)
# # th.save({'dd':dd, 'ee':ee, 'ev':ev}, 'data.p')
# d = th.load('data.p')
# d['ev'] = ev
# th.save(d, 'data.p')

# dd = th.load('data.p')
# ee = np.stack(dd['ev']).astype(np.int32)
# ec = ee[None, :, :] - ee[:, None, :]
# ec_sum = np.zeros([ec.shape[0], ec.shape[0]])
# for aa in (th.chunk(th.arange(10000), 20)):
#     ec_ = ec[:, :, aa]
#     ec_sum += np.abs(ec_).sum(-1)
# dd['ecv'] = ec_sum
# th.save(dd, 'data.p')


# root = '/home/ubuntu/ext_vol/inpca/inpca_results_all/'
# key = 'yh'
# fn = 'all_geod'

# didx = th.load(os.path.join(root, f"didx_geod_all.p")).reset_index(drop=True)
# f = h5py.File(os.path.join(root, f"w_{key}_{fn}.h5"), "r")
# w = f["w"][:]

# idxs = get_idx(didx, "opt=='sgd'")
# print(len(idxs))

# didx_sgd = didx.iloc[idxs].reset_index(drop=True)
# w_sgd = w[idxs, :][:, idxs]

# th.save(didx_sgd, os.path.join(root, f"didx_{fn}_sgd.p"))
# h = h5py.File(os.path.join(root, f"w_{fn}_sgd.h5"), 'w')
# dset = h.create_dataset('w', data=w_sgd)

# convert all loaded df probs to log form
# loc = '/home/ubuntu/ext_vol/inpca/results/models/loaded'
# all_files = glob.glob(os.path.join(loc, "*}.p"))[1819:]
# for f in tqdm.tqdm(all_files):
#     d = th.load(f)
#     yh = np.stack(d.yh)
#     yvh = np.stack(d.yvh)
#     if np.allclose(yh.sum(-1), 1):
#         print(f)
#         d['yh'] = np.array_split(np.log(yh), len(d))
#     if np.allclose(yvh.sum(-1), 1):
#         print(f)
#         d['yvh'] = np.array_split(np.log(yvh), len(d))
#     th.save(d, f)


# Redo distance to geodesic
# import json
# diffs = []
# cols = ['seed', 'aug', 'm', 'lr', 'opt', 'bs', 'wd']
# for r in tqdm.tqdm(range(0, 2300, 100)):
#     didx_geod = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/inpca_results/didx_yh_geod_c{r}.p')['dc']
#     ii_geod = didx_geod.groupby(cols).indices
#     for c in ii_geod.keys():
        
#         if c[2] == 'geodesic':
#             continue
#         dic = {'seed':int(c[0]), 'bseed':-1, 'aug':c[1], 'm':c[2],
#                     'bn':True, 'drop':0.0, 'opt':c[4], 'bs':int(c[5]), 'lr':float(c[3]), 'wd':float(c[6]),
#                     'corner':'normal', 'interp':False}
#         root = '/home/ubuntu/ext_vol/inpca/results/models/loaded/'
#         fn = json.dumps(dic).replace(' ', '') + '.p'
#         try:
#             d_saved = th.load(os.path.join(root, fn))[didx_geod.columns]
#         except FileNotFoundError:
#             print(fn)
        
#         if not (didx_geod.iloc[ii_geod[c]].reset_index(drop=True) == d_saved).all().all():
#             diffs.append((r,c))
#             print(r, c)
# th.save(diffs, 'diffs.p')

