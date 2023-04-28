import glob
import json
import os
import torch as th
import pandas as pd

loc = 'results/models/new'

r = []
for f in glob.glob(os.path.join(loc, '*')):

    k = json.loads(f[f.find('{'):f.find('}')+1])
    if 'aug' not in k.keys():
        k['aug'] = False
    k['bs'] = 200
    fn = json.dumps(k).replace(' ', '') + '.p'
    fn = os.path.join(loc, fn)
    os.rename(f, fn)
