from setup import *
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

@call_parse
def main(
    test: Param('test', bool, default=False),
    perm: Param('perm', bool, default=False)
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

    for f in features:
        le = LabelEncoder()
        dcross[f] = le.fit_transform(dcross[f])

    x = np.stack(dcross[features+['d2geod_x', 'd2geod_y']].values)
    model = RandomForestRegressor()
    model.fit(x, dcross['dcross'])
    model.score(x, dcross['dcross'])

    importances = model.feature_importances_
    indices = np.argsort(importances)

    f, ax = plt.subplots()
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    _ = ax.set_yticklabels(np.array(features + ['d2geod_x', 'd2geod_y'])[indices])

    f.savefig(f'../plots/{fn}_feature_importance.pdf', bbox_inches='tight')

    if perm:
        from sklearn.inspection import permutation_importance
        features = features + ['d2geod_x', 'd2geod_y']
        permImp = permutation_importance(model, x, dcross['dcross'], n_repeats=100, 
                    random_state=0)
        data = pd.DataFrame(permImp['importances'].T, columns=features)

        data = pd.DataFrame(permImp['importances'].T, columns=features)
        f, ax = plt.subplots()
        indices = permImp['importances_mean'].argsort()
        sns.boxplot(data=data, orient='h', ax=ax)
        ax.set(xlim=(-0.02, 0.35))

        f.savefig(f'../plots/{fn}_feature_permutation_importance.pdf', bbox_inches='tight')
