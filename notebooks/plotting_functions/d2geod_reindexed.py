from setup import *


@call_parse
def main(test: Param('test', bool, default=False)):

    key = 'yvh' if test else 'yh'
    fn = 'test' if test else 'train'


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

    # get distance to geodesic
    key = 'bs'
    choices = [200]

    geod_idx = list(didx[didx.m == 'geodesic'].index)

    idx = list(didx[didx[key].isin(choices)].index) + geod_idx
    d = didx.iloc[idx].reset_index(drop=True)
    d = d[d.m != 'geodesic'].reset_index(drop=True)
    w = dists[:, :, idx][:, idx, :]
    dist2geod = w[:, :-1, -1].T
    d['dist_to_geod'] = np.array_split(dist2geod, len(d))
    d = d.explode('dist_to_geod').explode('dist_to_geod')
    d['t'] = np.tile(np.arange(100), len(dist2geod))

    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.lineplot(data=d[d.seed > 0].reset_index(), x='t',
                    y='dist_to_geod', hue='m', style='opt', lw=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title(f"{key}={choices}")
    f.savefig(f'../plots/all_models_{fn}_d2geod.pdf', bbox_inches='tight')
