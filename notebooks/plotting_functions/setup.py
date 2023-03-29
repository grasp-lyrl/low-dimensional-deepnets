import sys
root = '/home/ubuntu/ext_vol'
sys.path.insert(0, f'{root}/inpca')
import pandas as pd
import numpy as np
import torch as th
from fastcore.script import *
import scipy.sparse.linalg as sp
import seaborn as sns
import matplotlib.pyplot as plt
from utils import triplot, CDICT_M, get_idx, plot_dendrogram
from embed import proj_
import h5py
plt.rcParams['figure.figsize'] = [4, 4]
plt.rcParams['figure.dpi'] = 200
sns.set(context='notebook',
        style='ticks',
        font_scale=1,
        rc={'axes.grid': True,
            'grid.color': '.9',
            'grid.linewidth': 0.75})
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
