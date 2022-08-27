from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

def plot3d(dc, r, dims=[0, 1, 2], key='widen', markers=["o", "x", "s", "*", "+"],
           cmap='vlag', cdict=None, ckey='', sdict={}):
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    ee = r['e']
    print(ee)
    c = dc[ckey]
    if cdict:
        c = c.fillna('nan')
        c = c.map(cdict)
        print(cdict)
    d1, d2, d3 = dims
    for (i, (k, ii)) in enumerate(dc.groupby(key).indices.items()):
        xx = r['xp'][ii]
        s = sdict.get(k, 3)
        sc = ax.scatter(xx[:, d1], xx[:, d2], xx[:, d3],
                        c=c[ii], cmap=cmap, vmin=c.min(), vmax=c.max(),
                        label=k,  marker=markers[i], s=s, lw=0.5, alpha=0.5)

        #                 sc = ax.plot(xx[:, d1], xx[:, d2], label=k, lw=0.5)
        ax.set_xlabel(f'pc{d1}, {ee[d1]:.2f}')
        ax.set_ylabel(f'pc{d2}, {ee[d2]:.2f}')
        ax.set_zlabel(f'pc{d3}, {ee[d3]:.2f}')

    handles, labels = ax.get_legend_handles_labels()
    clb = plt.colorbar(sc, pad=0.2, ax=ax)
    clb.ax.set_title(ckey)
#     ax.legend(handles, labels, loc = 'lower center')


def triplot(dc, r, d=3, key='widen', cmap='vlag', cdict=None, ckey='', sdict={},
            markers=["o", "x", "s", "*", "+"],
            evals=False, plot_avg=False, avggroupby=['opt'], plot_lines=False):

    figs, axs = plt.subplots(d-1, d-1, figsize=(10, 10),
                             sharex=True, sharey=True)
    ee = r['e']
    c = dc[ckey]
    if cdict:
        c = c.fillna('nan')
        c = c.map(cdict)
        print(cdict)
    for (i1, d1) in enumerate(range(d-1)):
        for (i2, d2) in enumerate(range(d1+1, d)):
            ax = axs[d2-1, d1]
            ax.axis("square")
            for (i, (k, ii)) in enumerate(dc.groupby(key).indices.items()):
                s = sdict.get(k, 3)
                xx = r['xp'][ii]
                sc = ax.scatter(xx[:, d1], xx[:, d2],
                                c=c[ii], cmap=cmap, vmin=c.min(), vmax=c.max(),
                                label=k,  marker=markers[i], s=s, lw=0.5, alpha=0.5)

                if plot_lines:
                    config_cols = ['seed', 'm', 'opt', 'bs', 'aug', 'bn', 'lr', 'wd']
                    config_cols.remove(key)
                    dc_ = dc.iloc[ii]
                    for (c, ii_) in dc_.groupby(config_cols).indices.items():
                        xx_ = xx[ii_]
                        print(dc_.iloc[ii_]['t'])
                        _ = sns.lineplot(x=xx_[:, d1], y=xx_[:, d2], ax=ax, legend=False, lw=0.2)

                    #                 sc = ax.plot(xx[:, d1], xx[:, d2], label=k, lw=0.5)
                if ee[d1] < 0:
                    ax.spines['bottom'].set_color('red')
                if ee[d2] < 0:
                    ax.spines['left'].set_color('red')
                ax.set_xlabel(f'pc{d1}')
                ax.set_ylabel(f'pc{d2}')
            if plot_avg:
                iavg = get_idx(dc, f't >= 0 & seed == -1')
                for (k, v) in dc.iloc[iavg].groupby(avggroupby).indices.items():
                    avg_ms = r['xp'][v]
                    avg = sns.lineplot(x=avg_ms[:, d1], y=avg_ms[:, d2],
                                       label=k, ax=ax, legend=False, lw=1.0)
    handles, labels = ax.get_legend_handles_labels()
    if evals:
        ax = figs.add_subplot(d-1, d-1, 2)
        ax.set_yscale('log')
        ax.grid()
        w = 0.5
        for e in ee:
            ax.plot((-w/4, w/4), (np.abs(e), np.abs(e)),
                    c='k' if e > 0 else 'r')
        ax.set_xlim([-w/2, w/2])
        ax.get_xaxis().set_visible(False)

    ax = axs[0, -1]
    clb = plt.colorbar(sc, pad=0.2, ax=ax)
    clb.ax.set_title(ckey)
    ax = axs[0, -1]
    ax.legend(handles, labels, loc='lower center')


def main():
    choices = {
        'm': ["allcnn", "convmixer", "fc",
              "vit", "wr-10-4-8", "wr-16-4-64"],
        'opt': ["adam", "sgd", "sgdn"],
        'lr' : [0.001, 0.1, 0.25, 0.5, 0.0005, 0.0025, 0.00125,
     0.005],
        'bs': [200, 500],
        'aug': ['simple', 'none'],
        'wd': [0., 1.e-03, 1.e-05]
    }

    loc = 'inpca_results_all/47'
    fn = 'yh_all'
    save_fn = "selected"
    key = 'm'
    ckey = 't'
    cmap = 'vlag'
    if ckey in choices.keys():
        cdict = {c:i for (i, c) in enumerate(choices[ckey])}
    else:
        cdict = None
    plot_3d = False 
    tri_plot = True
    plot_lines = True

    sns.set_style('whitegrid')

    _, dc = th.load(os.path.join(loc, 'didx_%s.p' % fn))
    dc = dc.reset_index()
    r = th.load(os.path.join(loc, 'r_%s.p' % fn))
    dc = dc.drop("index", axis=1)
    for (config, idxs) in dc.groupby(['m', 'opt', 'bs', 'aug', 'lr', 'wd']).indices.items():
        tmax = dc.iloc[idxs]['t'].max()
        dc.loc[idxs, 't'] /= tmax

    ii = get_idx(dc, ' and '.join(f'{k} in {v}' for (k, v) in choices.items()))
    dc = dc.iloc[ii].reset_index()
    r['xp'] = r['xp'][ii]
    r['v'] = r['v'][ii, :]
    if plot_3d:
        plot3d(dc, r, dims=[0, 1, 2], key=key,
            ckey=ckey, cmap=cmap, cdict=cdict, markers=["o", "x", "s", ".", "*", "+"])
        plt.savefig(os.path.join(loc, f'{save_fn}_3d.png'), dpi=400)
    if tri_plot:
        triplot(dc, r, d=4, key=key, ckey=ckey, cmap=cmap, cdict=cdict,
                markers = ["o", "x", "s", ".", "*", "+"], plot_lines=plot_lines,
                evals=True) 
        plt.savefig(os.path.join(loc, 'plots', f'{save_fn}_triplot.png'), dpi=400)

if __name__ == '__main__':
    main()
