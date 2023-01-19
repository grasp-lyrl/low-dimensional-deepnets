import os
import h5py
import numpy as np
import torch as th
import time

fn = "yh_all"
save_fn = "yh_all"
root = "/home/ubuntu/ext_vol/inpca/inpca_results_avg_new"

print("loading w")
start_t = time.time()
f = h5py.File(os.path.join(root, f"w_{fn}.h5"), "r")
w = f["w"][:]
print("w loaded, t: ", time.time() - start_t)
start_t = time.time()
print(w.shape)

print("centering")
n = w.shape[0]
dmean = w.mean(0, keepdims=True)
w -= dmean
w -= w.mean(1, keepdims=True)
w *= -0.5
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
