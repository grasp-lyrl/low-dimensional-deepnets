import numpy as np
import torch as th

import os, pdb, sys, json, glob, tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import h5py
from utils import c2fn, dbhat, dinpca, lazy_embed, load_d, full_embed
from reparameterization import v0

# all analysis in this file excludes points that are far away from the main manifold

g = th.load(
    '/home/ubuntu/ext_vol/inpca/results/models/loaded/{"seed":0,"bseed":-1,"aug":"na","m":"geodesic","bn":"na","drop":"na","opt":"geodesic","bs":"na","lr":"na","wd":"na","interp":false}.p')
geod = np.stack(g.yh)[::2, :, :]
p0 = geod[0, :]
ps = geod[-1, :]
N, C = p0.shape


def num_deriv(yhs, center=5, win=5):
    # yhs = [num_models, num_times, num_samples, num_classes]
    yhs = yhs[:, center-win:center+win+1, :, :]
    tan = (yhs[:, 2*win:win:-1, :, :] - yhs[:, :win, :, :])

    tan = tan/np.arange(win*2, 0, -2)[None, :, None, None]
    return tan.sum(1) / (win-1)

def sph_interp(init, tan, ts):
    pts = np.stack([np.sqrt(init)+t*tan for t in ts])
    pts = pts ** 2
    pts /= pts.sum(-1, keepdims=True)
    return pts


def lin_interp(init, tan, ts):
    pts = np.stack([init+t*tan for t in ts])
    pts = np.abs(pts)
    pts /= pts.sum(-1, keepdims=True)
    return pts


def avg_v0(p, q, center=0, win=5):
    # p is a fixed point, shape=(num_models, 1, num_samples, num_classes)
    # q.shape = (num_models, num_t, num_samples, num_classes)
    vt = np.stack([v0(p, q[:, center+i, :]) for i in range(win)])
    return vt.mean(0)


def get_all_tans(yhs_, p0, ps, it=40, et=40):
    # sampling tangent vectors on sphere
    nmodels, T, _, _ = yhs_.shape
    vinit = avg_v0(
               p=np.sqrt(np.tile(p0[None, :, :], [nmodels, 1, 1])),
               q=np.sqrt(yhs_), center=0, win=5
              )
    vinit_avg = vinit.mean(0)
    vinit_avg /= np.linalg.norm(vinit_avg)
    init_tan = sph_interp(p0, vinit_avg, ts=np.arange(it)).squeeze()

    vend = avg_v0(p=np.sqrt(np.tile(ps[None, :, :], [nmodels, 1, 1])),
                   q=np.sqrt(yhs_), center=T-10, win=3
                  )
    vend_avg = vend.mean(0)
    vend_avg /= np.linalg.norm(vend_avg)
    end_tan = sph_interp(ps, -vend_avg, ts=np.arange(et)).squeeze()
    return init_tan, end_tan


def get_avg_pts(lam1, lam2, n1, n2, configs, chunks=100, std=0.05):
    Z1, Z2 = 0, 0

    avg1 = np.zeros([N, C])
    avg2 = np.zeros([N, C])
    for i in range(0, len(configs), chunks):
        fs = [c2fn(c) for c in configs[i:i+chunks]]
        d = load_d(fs, avg_err=True)
        d = d.reset_index(drop=True)
        d = d[d.favg < 4].reset_index(drop=True)

        yhs = np.stack(d.yh)
        assert (yhs >= 0).all()
        w1 = np.exp(-(d.lam_yh.values-lam1)**2/(2*std**2))
        w2 = np.exp(-(d.lam_yh.values-lam2)**2/(2*std**2))
        Z1 += w1.sum()
        Z2 += w2.sum()

        avg1 += (w1.reshape(-1, 1, 1)*yhs).sum(0)
        avg2 += (w2.reshape(-1, 1, 1)*yhs).sum(0)

    avg1, avg2 = avg1/Z1, avg2/Z2
    init_pts = p0 + np.linspace(0, 1, n1)[:, None, None] * np.repeat(
        (avg1-p0)[None, :], n1, axis=0) if n1 > 0 else avg1[None, :]
    end_pts = ps + np.linspace(0, 1, n2)[:, None, None] * np.repeat(
        (avg2-ps)[None, :], n2, axis=0) if n2 > 0 else avg2[None, :]
    return init_pts, end_pts


def get_embed_four_pts(lam1, lam2, n1, n2, fn='', root='/home/ubuntu/ext_vol/inpca/inpca_results_all/tangents'):
    fn = f'{fn}{lam1}-{lam2}-{n1}-{n2}'
    chunks = 100
    didx = th.load(
        '/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all_progress.p')
    didx = didx[didx.m != 'geodesic'].reset_index(drop=True)
    cols = ["seed", "m", "opt", "bs", "aug", "lr", "wd"]
    configs = list(didx.groupby(cols, sort=False).count().index)
    init_pts, end_pts = get_avg_pts(lam1, lam2, n1, n2, configs)
    all_points = np.stack([p0, ps, init_pts, end_pts])
    th.save(all_points, os.path.join(root, f'all_points_{fn}.p'))

    dists = dbhat(all_points, all_points, chunks=100)
    dmean, r = full_embed(dists)
    th.save(r, os.path.join(root, f'r_tangent_{fn}.p'))

    for i in range(0, len(configs), chunks):
        fs = [c2fn(c) for c in configs[i:i+chunks]]
        d = load_d(fs, avg_err=True)
        d = d.reset_index(drop=True)
        d = d[d.favg < 4].reset_index(drop=True)
        yhs = np.stack(d.yh)
        pts = lazy_embed(new_pts=th.tensor(yhs),
                         ps=all_points.float(),
                         evals=r['e'], evecs=r['v'], d_mean=dmean, chunks=500)
        xp = np.vstack([r['xp'], pts])
        r['xp'] = xp
        th.save(r, os.path.join(root, f'r_tangent_{fn}.p'))
    return fn


def get_rel_err(fn, ne=4, didx_fn='', rtrue_fn='', root='/home/ubuntu/ext_vol/inpca/inpca_results_all/tangents'):
    r = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/r_tangent_{fn}.p')

    if didx_fn:
        d = th.load(os.path.join(root, f'didx_{didx_fn}.p'))
    else:
        d = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_no_outliers.p')

    if rtrue_fn:
        rtrue = th.load(os.path.join(root, f'r_tangent_{rtrue_fn}.p'))
    else:
        rtrue = th.load(
            '/home/ubuntu/ext_vol/inpca/inpca_results_all/r_yh_no_outliers.p')

    start_idx = len(r['xp']) - len(rtrue['xp'])
    ii = np.array(d['index'])
    rel_err = []

    for i in range(1, 1+ne):
        wsum = 0
        tan_err = 0
        true_err = 0
        for aa in tqdm.tqdm(th.chunk(th.arange(len(d)), 20)):
            for bb in th.chunk(th.arange(len(d)), 20):
                dd = dinpca(th.tensor(r['xp'][start_idx+aa, :i]),
                            th.tensor(r['xp'][start_idx+bb, :i]),
                            dev='cuda',
                            sign=th.tensor(np.sign(r['e'][:i])).double()).cpu().numpy()
                if rtrue_fn:
                    dtrue = dinpca(th.tensor(rtrue['xp'][aa, :i]), 
                                   th.tensor(rtrue['xp'][bb, :i]), 
                                   sign=th.tensor(np.sign(rtrue['e'][:i])).double()).cpu().numpy()
                else:
                    dtrue=0
                with h5py.File(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/w_yh_all_geod.h5', 'r') as f:
                    amin, amax = ii[aa].min(), ii[aa].max()
                    bmin, bmax = ii[bb].min(), ii[bb].max()
                    w = f['w'][amin:amax+1, :][:, bmin:bmax+1]
                    iia = np.arange(
                        amax-amin+1)[np.isin(np.arange(amin, amax+1), ii[aa])]
                    iib = np.arange(
                        bmax-bmin+1)[np.isin(np.arange(bmin, bmax+1), ii[bb])]
                    w = np.array(w)[iia, :][:, iib]
                tan_err += np.abs(dd.T-w).sum()
                true_err += np.abs(dtrue.T-w).sum()
                wsum += w.sum()
        rel_err.append([tan_err/wsum, true_err/wsum])
        th.save(rel_err, os.path.join(root, f'rel_err_{fn}.p'))


if __name__ == '__main__':

    fn = get_embed_four_pts(0.2, 0.8, 0, 0)
    get_rel_err(fn = fn, ne=4)
