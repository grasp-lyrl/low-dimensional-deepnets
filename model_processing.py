import os
import numpy as np
import torch as th
from utils import load_d, avg_model, interpolate

models = ["wr-4-8", "allcnn-96-144", "fc-1024-512-256-128"]
opts = ["adam", "sgdn", "sgd"]
T = 45000
ts = []
for t in range(T):
    if t < T//10:
        if t % (T//100) == 0:
            ts.append(t)
    else:
        if t % (T//10) == 0 or (t == T-1):
            ts.append(t)
ts = np.array(ts)
tmap = {i:ts[i] for i in range(len(ts))}
pts = np.concatenate([np.arange(ts[i], ts[i+1], (ts[i+1]-ts[i]) // 5) for i in range(len(ts)-1)])
N = len(pts)

def main():
    loc = 'results/models/new'
    d = load_d(loc, 
               cond={'bs':[200], 'aug':[True], 'wd':[0.0], 'bn':[True], 'm':models, 'opt':opts},
               numpy=True, probs=True,
               avg_err=True, drop=True)
    d['t'].map(tmap)
    d = avg_model(d, groupby=['m', 'opt', 't'], probs=True, get_err=True, 
                  update_d=True, compute_distance=False, dev='cuda')['d']

    d = interpolate(d, ['seed', 'm', 'opt', 'avg'], ts, pts, keys=[
                     'yh', 'yvh'], dev='cuda')

    th.save(d, os.path.join(loc, "all_models_interp.p"))

if __name__ == "__main__":
    main()
