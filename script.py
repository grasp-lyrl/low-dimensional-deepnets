import h5py
from embed import *
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

##############################
# compute euclidean distance #
##############################

# didx = th.load(
#     "/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all.p").reset_index(drop=True)
# idxs = get_idx(
#     didx, "(m=='allcnn' and aug == 'simple' and wd==0.0) or m=='geodesic'")
# didx = didx.iloc[idxs].reset_index(drop=True)
# cols = ['seed', 'm', 'opt', 'bs', 'aug', 'lr', 'wd']
# idxs = didx.groupby(cols).indices
# fs = []
# root = '/home/ubuntu/ext_vol/inpca/results/models/loaded'
# for c in idxs.keys():
#     seed, m, opt, bs, aug, lr, wd = c
#     seed = int(seed)
#     bs = int(bs) if bs != 'na' else bs
#     if bs == 'na':
#         fn = json.dumps(dict(seed=seed, bseed=-1, aug='na', m=m, bn='na', drop='na',
#                                 opt=opt, bs=bs, lr=lr, wd=wd, interp=False)).replace(" ", "")
#     else:
#         fn = json.dumps(dict(seed=seed, bseed=-1, aug=aug, m=m, bn=True, drop=0.0,
#                                 opt=opt, bs=bs, lr=lr, wd=wd, corner='normal', interp=False)).replace(" ", "")
    
#     fs.append(os.path.join(root, f'{fn}.p'))

# d1 = load_d(
#     file_list=fs,
#     avg_err=False,
# )

# d2 = None
# key = "yh"
# fn = "euclid"
# loc = "inpca_results_all"
# idx = ["seed", "m", "opt", "t", "err", "verr", "bs", "aug", "lr", "wd"]
# xembed(
#     d1,
#     d2,
#     fn=f"{key}_{fn}",
#     probs=True,
#     key=key,
#     loc=loc,
#     idx=idx,
#     force=True,
#     distf="deuclid",
#     reduction="none",
#     chunks=4000,
#     proj=True,
#     save_didx=True,
# # )
# w = th.load("/home/ubuntu/ext_vol/inpca/inpca_results_all/w_yh_euclid.p")
# w = (w**2) / 25000
# l = np.eye(w.shape[0]) - 1.0 / w.shape[0]
# w = -l @ w @ l / 2
# r = proj_(w, w.shape[0], 3)
# th.save(r, '/home/ubuntu/ext_vol/inpca/inpca_results_all/r_yh_euclid_mean.p')

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


root = '/home/ubuntu/ext_vol/inpca/inpca_results_all/'
key = 'yh'
fn = 'all_geod'

didx = th.load(os.path.join(root, f"didx_geod_all.p")).reset_index(drop=True)
f = h5py.File(os.path.join(root, f"w_{key}_{fn}.h5"), "r")
w = f["w"][:]

idxs = get_idx(didx, "opt=='sgd'")
print(len(idxs))

didx_sgd = didx.iloc[idxs].reset_index(drop=True)
w_sgd = w[idxs, :][:, idxs]

th.save(didx_sgd, os.path.join(root, f"didx_{fn}_sgd.p"))
h = h5py.File(os.path.join(root, f"w_{fn}_sgd.h5"), 'w')
dset = h.create_dataset('w', data=w_sgd)
