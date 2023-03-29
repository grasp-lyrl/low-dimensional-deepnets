from setup import *


@call_parse
def main(test: Param('test', bool, default=False)):

    key = 'yvh' if test else 'yh'
    fn = 'test' if test else 'train'
    cols = ['m', 'opt', 'bs', 'lr', 'wd', 'aug']

    dists = th.load(f'{root}/inpca/inpca_results_avg_new/dists_3d_{key}.p')
    didx = th.load(
        f'{root}/inpca/inpca_results_avg_new/didx_3d_{key}.p').reset_index(drop=True)

    # remove untrained models
    idxs = didx[didx.err <= 0.1].index
    didx = didx.iloc[idxs].reset_index(drop=True)
    dists = dists[:, :, idxs][:, idxs, :]

    # keep only Euclidean averages 
    idxs = didx[didx.seed >=-1].index
    didx = didx.iloc[idxs].reset_index(drop=True)
    dists = dists[:, :, idxs][:, idxs, :]

    # get distance to avg model
    all_d = pd.DataFrame()
    for (c, ii) in didx.groupby(cols).indices.items():
        avg_ii = ii[-1]
        m_ii = ii[:-1]
        d2avg = dists[:, m_ii, avg_ii]
        df = pd.DataFrame(d2avg)
        df['t'] = list(np.linspace(0, 1, 100))
        df = df.melt(id_vars='t')
        df['config'] = [tuple(c) for i in range(len(df))]
        all_d = pd.concat([all_d, df])

    for (i, c) in enumerate(cols):
        all_d[c] = np.stack(all_d.config)[:, i]

    # Boxplot w.r.t. progress
    nticks = 5
    step = (100 // nticks)-1
    f, ax = plt.subplots()
    ax = sns.boxplot(data=all_d[all_d.t.isin(all_d.t.unique()[
                    ::step])].reset_index(), x='t', y='value', color='b')
    ax.set(ylabel='Distance to mean trajectory')
    ax.set(xlabel='Progress')
    ticks = [i.get_text() for i in ax.get_xticklabels()]
    ticks = ax.set_xticks(np.arange(0, len(ticks)),
                        all_d.t.unique()[::step].round(1))
    f.savefig(f'../plots/all_models_{fn}_tube_width.pdf', bbox_inches='tight')
