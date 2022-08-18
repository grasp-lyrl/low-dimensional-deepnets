from utils import *
import torch as th
import numpy as np
import pandas as pd

def main():
    data = get_data()
    y, yv = th.tensor(data['train'].targets).long(), th.tensor(data['val'].targets).long()
    y_ = np.zeros((len(y), y.max()+1))
    y_[np.arange(len(y)), y] = 1
    yv_ = np.zeros((len(yv), yv.max()+1))
    yv_[np.arange(len(yv)), yv] = 1
    extra_pts = [dict(seed=0, m='true', t=th.inf, err=0., verr=0., yh=y_, yvh=yv_), 
                dict(seed=0, m='random', t=0, err=0.9, verr=0.9, 
                yh=np.ones([len(y), y.max()+1])/(y.max()+1), 
                yvh=np.ones([len(yv), yv.max()+1])/(yv.max()+1))
                ]
    extra_pts = pd.DataFrame(extra_pts)


if __name__ == "__main__":
    main()
