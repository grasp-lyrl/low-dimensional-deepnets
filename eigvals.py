import os
import h5py
import numpy as np
import torch as th
import time
from utils import get_idx
from fastcore.script import *

def main(key="yh",
         load_fn="all_with_normal", 
         save_fn="noinit_with_normal", 
         centering='normal',
         cond="(isinit == False) or (corner == 'normal')",
         cond_didx="/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/didx_all_with_normal.p",
         root="/home/ubuntu/ext_vol/inpca/inpca_results_all/corners"):

     if cond:
          didx = th.load(cond_didx)
          ii = get_idx(didx, cond)
          th.save(didx.iloc[ii].reset_index(drop=True), os.path.join(root, f"didx_{save_fn}.p"))

     wf = os.path.join(root, f"w_{key}_{load_fn}.h5")
     print("loading w from", wf)
     start_t = time.time()
     f = h5py.File(wf, "r")
     if cond:
          w = f["w"][ii, :][:, ii]
     else:
          w = f["w"][:]
     print("w loaded, t: ", time.time() - start_t)
     start_t = time.time()
     print(w.shape)

     print("centering")
     if centering == 'normal':
          n = w.shape[0]
          dmean = w.mean(0, keepdims=True)
          w = w - dmean
          w = w - w.mean(1, keepdims=True)
          w = w*(-0.5)
     elif centering == 'weighted':
          weight = np.ones(len(didx))
          iw = get_idx(didx, "m=='geodesic'")
          weight[iw] = len(ii)
          weight = weight[ii]
          weight /= weight.sum()
          dmean = (w*weight).sum(1, keepdims=True)
          w -= dmean
          w -= (w*weight[:, None]).sum(0)
          w *= -0.5
          w *= weight[None, :]
     elif centering == 'pca':
          w *= w
          dmean = w[:, :1]
          w = dmean + dmean.T - w
          w *= 0.5
     print("centered, t: ", time.time() - start_t)

     start_t = time.time()
     print("getting evals")
     import scipy.sparse.linalg as sp
     ne = 500
     es, vs = sp.eigsh(w, ne, which='LM', return_eigenvectors=True)

     r = dict(es=es, vs=vs, tr=np.trace(w), w_mean=dmean)
     th.save(r, os.path.join(root, f"r_{key}_{save_fn}.p"))
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
     th.save(r, os.path.join(root, f"r_{key}_{save_fn}.p"))
     print("projected, t: ", time.time() - start_t)
     start_t = time.time()

if __name__ == '__main__':
     # main(fn='yh_all_geod', save_fn='yh_allcnn', cond="m=='allcnn'", root='/home/ubuntu/ext_vol/inpca/inpca_results_all')
    fn = "all_euclid"
    for k in ['yh', 'yvh']:
        main(key=k, load_fn=fn, save_fn=fn, cond='',
             centering='pca',
             root="/home/ubuntu/ext_vol/inpca/inpca_results_all/euclidean")