from setup import *

key = 'yvh'
t = 100
mode = 'mean' # ['mean', 'snapshot']


# Load distances, indices
dists = th.load(f'{root}/inpca/inpca_results_avg_new/dists_3d_{key}.p')
didx = th.load(
    f'{root}/inpca/inpca_results_avg_new/didx_3d_{key}.p').reset_index(drop=True)

# Exclude avg models
non_avg = get_idx(didx, "seed >= 0")
didx = didx.iloc[non_avg].reset_index(drop=True)
dists = dists[:, :, non_avg][:, non_avg, :]

# Exclude models that did not train
idxs = didx[didx.err <= 0.1].index
didx = didx.iloc[idxs].reset_index(drop=True)
dists = dists[:, :, idxs][:, idxs, :]

# Compute average pairwise distance for each configuration
cols = ['m', 'opt', 'bs', 'lr', 'wd', 'aug']
idx = didx.groupby(cols).indices
pdists = np.zeros([100, len(idx), len(idx)])
keys = list(idx.keys())
for i in range(len(idx)):
    for j in range(i, len(idx)):
        ii = idx[keys[i]]
        jj = idx[keys[j]]
        dij = dists[:, ii, :][:, :, jj]
        pdists[:, i, j] = pdists[:, j, i] = dij.mean(axis=(1, 2))

rows = columns = np.array(keys, dtype="O")

# Accumulate distance w.r.t. time or take snapshot at time t
if mode == 'mean':
    avg_pdists = pdists[:t, :, :].sum(0)/100
elif mode == 'snapshot':
    avg_pdists = pdists[t, :, :]
np.fill_diagonal(avg_pdists, 0)
v = squareform(avg_pdists)

# Compute linkage
linkage = sch.linkage(v, method='complete', optimal_ordering=True)
cs = ['-'.join(str(l)) for l in columns.astype(str)]
rs = ['-'.join(str(l)) for l in rows.astype(str)]

ig = np.where(columns[:, 0] == 'geodesic')[0]
fig, dend = plot_dendrogram(linkage, columns,  didx=didx, key=key, 
                            above_threshold_color='k',
                            cdict=CDICT_M,
                            cols=cols,
                            color_by=0,
                            color_threshold=0)

fig.savefig(f'../plots/{key}_dend_{t}.pdf', bbox_inches='tight')
