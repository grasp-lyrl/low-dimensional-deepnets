import numpy as np
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

def main():
    loc = 'results/models/new'
    d = load_d(loc, 
               cond={'bs':[200], 'aug':[True], 'wd':[0.0], 'bn':[True], 'm':models, 'opt':opts},
               numpy=True, probs=False,
               avg_err=True, drop=True)

    d = avg_model(d, groupby=['m', 'opt', 't'], probs=False, get_err=True, 
                  update_d=True, compute_distance=False, dev='cuda')

    d = interpolate(d, columns=['m', 'opt', 'avg', 'seed'])

if __name__ == "__main__":
    main()