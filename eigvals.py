import os
import h5py
import numpy as np
import torch as th
import time
from utils import get_idx
from fastcore.script import *

def main(fn="yh_all", save_fn="yh_lam", 
          cond='lam_yh > 0.3',
          cond_didx="/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/didx_all.p",
         root="/home/ubuntu/ext_vol/inpca/inpca_results_all/corners"):
     if cond:
          didx = th.load(cond_didx)
          ii = get_idx(didx, cond)
          th.save(didx.iloc[ii].reset_index(drop=True), os.path.join(root, f"didx_{save_fn}.p"))

    # weight = np.ones(len(didx))
    # # iw = get_idx(didx, "m=='geodesic'")
    # # print(len(ii), len(iw))
    # # weight[iw] = len(ii)
    # # weight = weight[ii]
    # weight /= weight.sum()
    # print(weight.max(), weight.min())

     print("loading w")
     start_t = time.time()
     f = h5py.File(os.path.join(root, f"w_{fn}.h5"), "r")
     print(os.path.join(root, f"w_{fn}.h5"))
     if cond:
          w = f["w"][ii, :][:, ii]
     else:
          w = f["w"][:]
     print("w loaded, t: ", time.time() - start_t)
     start_t = time.time()
     print(w.shape)

     print("centering")
     n = w.shape[0]

    # dmean = (w*weight).sum(1, keepdims=True)
    # w -= dmean
    # w -= (w*weight[:, None]).sum(0)
    # w *= -0.5

    # w *= weight[None, :] 

     dmean = w.mean(0, keepdims=True)
     w = w - dmean
     w = w - w.mean(1, keepdims=True)
     w = w*(-0.5)
     print("centered, t: ", time.time() - start_t)
     start_t = time.time()

     print("getting evals")
     import scipy.sparse.linalg as sp
     ne = 500
     es, vs = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)

     r = dict(es=es, vs=vs, tr=np.trace(w), w_mean=dmean)
     th.save(r, os.path.join(root, f"r_{save_fn}.p"))
     print("t: ", time.time()-start_t)

     print("projecting")
     ne = 7
     e, v = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)
     ii = np.argsort(np.abs(e))[::-1]
     e, v = e[ii], v[:, ii]

     # scaling_factor = np.linalg.norm(v*np.sqrt(weight[:, None]), axis=0)
     # xp = v*np.sqrt(np.abs(e)) / scaling_factor
     xp = v*np.sqrt(np.abs(e))
     r.update(dict(xp=xp, e=e, v=v, diag=np.diag(w), fn=(w**2).sum().sum()))
     th.save(r, os.path.join(root, f"r_{save_fn}.p"))
     print("projected, t: ", time.time() - start_t)
     start_t = time.time()

if __name__ == '__main__':
     # main(fn='yh_all_geod', save_fn='yh_allcnn', cond="m=='allcnn'", root='/home/ubuntu/ext_vol/inpca/inpca_results_all')
    main()