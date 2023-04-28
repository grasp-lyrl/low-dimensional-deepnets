from setup import *
from autogluon.tabular import TabularDataset, TabularPredictor

@call_parse
def main(
    test: Param('test', bool, default=False),
    perm: Param('perm', bool, default=False),
    t: Param('t', int, default=100)
    ):

    key = 'yvh' if test else 'yh'
    fn = 'test' if test else 'train'
    cols = ['m', 'opt', 'bs', 'lr', 'wd', 'aug']

    dists = th.load(f'{root}/inpca/inpca_results_avg_new/dists_3d_{key}.p')
    didx = th.load(
        f'{root}/inpca/inpca_results_avg_new/didx_3d_{key}.p').reset_index(drop=True)

    # remove avg models
    non_avg = get_idx(didx, "seed >= 0")
    didx = didx.iloc[non_avg].reset_index(drop=True)
    dists = dists[:, :, non_avg][:, non_avg, :]

    # remove untrained models
    idxs = didx[didx.err <= 0.1].index
    didx = didx.iloc[idxs].reset_index(drop=True)
    dists = dists[:, :, idxs][:, idxs, :]

    idx = didx.groupby(cols).indices

    # compute average distance between trajectories
    pdists = np.zeros([100, len(idx), len(idx)])
    keys = list(idx.keys())
    for i in range(len(idx)):
        for j in range(i, len(idx)):
            ii = idx[keys[i]]
            jj = idx[keys[j]]
            dij = dists[:, ii, :][:, :, jj]
            pdists[:, i, j] = pdists[:, j, i] = dij.mean(axis=(1, 2))

    t = 100
    dt = pdists[:t, :, :].sum(0) / 100

    dd = pd.DataFrame(np.array(keys, dtype="O"), columns=cols)
    dd['d2geod'] = dt[dd[dd.m=='geodesic'].index[0], :]
    dcross = dd.merge(dd, how='cross')
    dcross['dcross'] = dt.ravel()

    features = ['m', 'opt', 'bs', 'aug', 'wd']
    for f in cols:
        dcross[f] = dcross[[f'{f}_x', f'{f}_y']].astype(
        str).apply(' '.join, axis=1)

    features = features +  ['d2geod_x', 'd2geod_y', 'dcross']
    dcross = dcross[features]
    agg_fn = {f:'first' for f in features}
    dcross = dcross.groupby(['dcross']).aggregate(agg_fn).reset_index(drop=True)


    train_data = TabularDataset(dcross)
    predictor = TabularPredictor(label='dcross').fit(train_data)
    train_importances = predictor.feature_importance(data=train_data)
    f, ax = plt.subplots()
    sns.boxplot(data=train_importances.loc[['m', 'bs','opt','aug','wd']].reset_index(), x='importance', y='index', orient='h', ax=ax)
    ax.set(ylabel='feature')
    f.savefig('../plots/feature_importance.pdf', bbox_inches='tight')
