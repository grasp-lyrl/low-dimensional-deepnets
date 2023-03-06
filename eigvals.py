import os
import h5py
import numpy as np
import torch as th
import time
from utils import get_idx

fn = "yvh_all_geod"
save_fn = "yvh_half_weighted"
root = "/home/ubuntu/ext_vol/inpca/inpca_results_all"
cond = "seed<=45" 

didx = th.load("/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all.p")
ii = get_idx(didx, cond)
th.save(didx.iloc[ii].reset_index(drop=True), os.path.join(root, f"didx_{save_fn}.p"))

weight = np.ones(len(ii))
weight[-1] = 600
weight /= weight.sum()

print("loading w")
start_t = time.time()
f = h5py.File(os.path.join(root, f"w_{fn}.h5"), "r")
w = f["w"][ii, :][:, ii]
print("w loaded, t: ", time.time() - start_t)
start_t = time.time()
print(w.shape)

print("centering")
n = w.shape[0]

dmean = (w*weight).sum(1, keepdims=True)
w -= dmean
w -= (w*weight[:, None]).sum(0)
w *= -0.5

# dmean = w.mean(0, keepdims=True)
# w -= dmean
# w -= w.mean(1, keepdims=True)
# w *= -0.5
print("centered, t: ", time.time() - start_t)
start_t = time.time()

print("getting evals")
import scipy.sparse.linalg as sp
ne = 50
es, vs = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)

r = dict(es=es, vs=vs, tr=np.trace(w), w_mean=dmean)
th.save(r, os.path.join(root, f"r_{save_fn}.p"))
print("t: ", time.time()-start_t)

print("projecting")
# import scipy.linalg as sp
ne = 7
e, v = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)
ii = np.argsort(np.abs(e))[::-1]
e, v = e[ii], v[:, ii]

xp = v*np.sqrt(np.abs(e))
r.update(dict(xp=xp, e=e, v=v, diag=np.diag(w), fn=np.linalg.norm(w, 'fro')))
th.save(r, os.path.join(root, f"r_{save_fn}.p"))
print("projected, t: ", time.time() - start_t)
start_t = time.time()
