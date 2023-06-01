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

#####################
# tangent embedding #
#####################

# from reparameterization import v0

# fn = 'four_pts_avg_0.1_0.9_0_0'
# get_rel_err_four_pts(fn)

# fn = 'all_tangents'


# def sph_interp(init, tan, ts):
#     pts = np.stack([np.sqrt(init)+t*tan for t in ts])
#     pts = pts ** 2
#     pts /= pts.sum(-1, keepdims=True)
#     return pts


# def lin_interp(init, tan, ts):
#     pts = np.stack([init+t*tan for t in ts])
#     pts = np.abs(pts)
#     pts /= pts.sum(-1, keepdims=True)
#     return pts


# def avg_v0(p, q, center=0, win=5):
#     # p is a fixed point, shape=(num_models, 1, num_samples, num_classes)
#     # q.shape = (num_models, num_t, num_samples, num_classes)
#     vt = np.stack([v0(p, q[:, center+i, :]) for i in range(win)])
#     return vt.mean(0)


# def embed_(dists, ne):
#     dmean = dists.mean(0)
#     l = np.eye(dists.shape[0]) - 1.0 / dists.shape[0]
#     dists = -l @ dists @ l / 2
#     r = proj_(dists, len(dists), ne)
#     return r


# def get_all_tans(yhs_, p0, ps, it=40, et=40):
#     # sampling tangent vectors on sphere
#     nmodels, T, _, _ = yhs_.shape
#     vinit = avg_v0(
#                p=np.sqrt(np.tile(p0[None, :, :], [nmodels, 1, 1])),
#                q=np.sqrt(yhs_), center=5, win=4
#               )
#     vinit_avg = vinit.mean(0)
#     vinit_avg /= np.linalg.norm(vinit_avg)
#     init_tan = sph_interp(p0, vinit_avg, ts=np.arange(it)).squeeze()

#     vend = avg_v0(p=np.sqrt(np.tile(ps[None, :, :], [nmodels, 1, 1])),
#                    q=np.sqrt(yhs_), center=T-15, win=5
#                   )
#     vend_avg = vend.mean(0)
#     vend_avg /= np.linalg.norm(vend_avg)
#     end_tan = sph_interp(ps, -vend_avg, ts=np.arange(et)).squeeze()
#     return init_tan, end_tan


# g = th.load(
#     '/home/ubuntu/ext_vol/inpca/results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na","interp":false}.p')
# geod = np.stack(g.yh)[::2, :, :]
# p0 = geod[0, :]
# ps = geod[-1, :]
# N, C = p0.shape

# def c2fn(c):
#     root = '/home/ubuntu/ext_vol/inpca/results/models/loaded/'
#     fdict = dict(seed=c[0], bseed=-1, aug=c[4], m=c[1], bn=True, drop=0., opt=c[2],
#                  bs=int(c[3]), lr=float(c[-2]), wd=float(c[-1]), corner='normal', interp=False)
#     fn = os.path.join(root, f'{json.dumps(fdict).replace(" ", "")}.p')
#     return fn

# chunks = 100
# didx = th.load(
#     '/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all_progress.p')
# didx = didx[didx.m != 'geodesic'].reset_index(drop=True)
# cols = ["seed", "m", "opt", "bs", "aug", "lr", "wd"]
# configs = list(didx.groupby(cols, sort=False).count().index)
# it, et = 50, 50
# ne = 2

# chunks=100
# vinit = np.zeros([N, C])
# vend = np.zeros([N, C])
# for i in range(0, len(configs), chunks):
#     fs = [c2fn(c) for c in configs[i:i+chunks]]
#     d = load_d(fs, avg_err=True)
#     d = d.reset_index(drop=True)
#     d = d[d.favg < 4].reset_index(drop=True)
#     cols = ["seed", "m", "opt", "bs", "aug", "lr", "wd"]
#     yhs_ = []
#     for ii in d.groupby(cols).indices.values():
#         y_ = np.squeeze(np.stack(d.iloc[ii].yh))
#         if (y_ < 0).any():
#             y_ = np.exp(y_)
#         yhs_.append(np.vstack([y_[:10], y_[-10:]]))
#     yhs_ = np.stack(yhs_) 
#     print(yhs_.shape)


#     nmodels, T, _, _ = yhs_.shape
#     vinit += avg_v0(
#             p=np.sqrt(np.tile(p0[None, :, :], [nmodels, 1, 1])),
#             q=np.sqrt(yhs_), center=5, win=4
#             ).sum(0)
#     vend += avg_v0(p=np.sqrt(np.tile(ps[None, :, :], [nmodels, 1, 1])),
#                 q=np.sqrt(yhs_), center=T-15, win=5
#                 ).sum(0)
#         vend += avg_v0(p=np.sqrt(np.tile(ps[None, :, :], [nmodels, 1, 1])),
#                     q=np.sqrt(yhs_), center=T-15, win=5
#                     ).sum(0)

# vinit_avg = vinit / len(configs)
# vinit_avg /= np.linalg.norm(vinit_avg)
# vend_avg = vend / len(configs)
# vend_avg /= np.linalg.norm(vend_avg)

# init_tan = sph_interp(p0, vinit_avg, ts=np.arange(50)).squeeze()
# end_tan = sph_interp(ps, -vend_avg, ts=np.arange(50)).squeeze()

# all_points = th.tensor(np.vstack([init_tan, geod, end_tan]))
# th.save(all_points, f'all_points_tangent.p')

# all_points = th.load('all_points_tangent.p')
# dists = dbhat(all_points, all_points, chunks=100)
# dmean = dists.mean(0)
# l = np.eye(dists.shape[0]) - 1.0 / dists.shape[0]
# dists = -l @ dists @ l / 2
# e, v = np.linalg.eigh(dists)
# ii = np.argsort(np.abs(e))[::-1]
# e, v = e[ii], v[:, ii]
# xp = v * np.sqrt(np.abs(e))
# r = dict(xp=xp, e=e, v=v)
# th.save(r, f'r_tangent_{fn}.p')

# chunks = 100
# for i in range(0, len(configs), chunks):
#     fs = [c2fn(c) for c in configs[i:i+chunks]]
#     d = load_d(fs, avg_err=True)
#     d = d.reset_index(drop=True)
#     d = d[d.favg < 4].reset_index(drop=True)
#     yhs = np.stack(d.yh)
#     pts = lazy_embed(new_pts=th.tensor(yhs),
#                      ps=all_points.float(),
#                      evals=r['e'], evecs=r['v'], d_mean=dmean, chunks=500)
#     xp = np.vstack([r['xp'], pts])
#     r['xp'] = xp
#     th.save(r, f'r_tangent_{fn}.p')


# fn = get_embed_four_pts(0.2, 0.8, 0, 0)
# get_rel_err_four_pts(fn = fn, n=4, didx_fn=fn, rtrue_fn=fn)
