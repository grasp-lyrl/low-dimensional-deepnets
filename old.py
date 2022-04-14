import torch as th
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from utils import *

def relabel_data(fn, y, frac=0.1, dev='cuda'):
    d = th.load(fn)
    yh = d[-1]['yh'].to(dev)
    _, yi = th.sort(yh, dim=1)
    y_new = yi[:, -2]
    # balanced relabel
    ss = int(len(yh)*frac//10)
    ys = th.chunk(th.arange(len(y)), 10)
    for yy in ys:
        idx = yy[th.randperm(len(yy))][:ss]
        y[idx] = y_new[idx]


def F_test(d, avg, dists, fixed='m', group='opt', key='yh', distf=th.cdist, ts=range(20),
           choices={'m': ["'wr-4-8'", "'allcnn-96-144'", "'fc-1024-512-256-128'"],
                    'opt': ["'adam'", "'sgdn'", "'sgd'"]}, verbose=False):
    F = []
    choices = choices[fixed]
    for c in choices:
        for t in ts:
            cond = f"{fixed} == {c} & t=={t}"
            allm = d.iloc[get_idx(d, cond)]
            levels = len(allm[group].unique())
            ni = len(allm) // levels

            om = th.Tensor(np.stack(allm[key]).mean(0))
            om = om.view(1, -1, 10).transpose(0, 1)

            gm = th.Tensor(np.stack(avg.iloc[get_idx(avg, cond)][key]))
            gm = gm.view(levels, -1, 10).transpose(0, 1)

            dms = distf(om, gm).mean(0)
            sb = dms.sum() * ni / (levels-1)
            
            sw = (dists.iloc[get_idx(dists, f"{cond} & key == '{key}'")]['dist'].values).sum(
            ) / (len(allm)-levels)
            if verbose:
                print('sb: ', sb, 'sw: ', sw)
            F.append({fixed: c.strip("''"), 'sb': sb.item(),
                     'sw': sw.item(), 'F': (sb/sw).item(), 't': t})
    return F
    

def avg_plot(dc, r, d=3, configs=["opt == 'adam'"], cmap='vlag', tri=False,
             mkey='', mdict=None, ckey='', cdict=None, evals=False):
    """_summary_
    Given projection, average over configs.
    """
    ee = r['e']
    print(ee)

    if tri:
        figs, axs = plt.subplots(
            d-1, d-1, figsize=(10, 10), sharex=True, sharey=True)
    else:
        fig = plt.figure(1, figsize=(10, 10))
        plt.clf()
        ax = fig.add_subplot(projection='3d')
    if len(configs) <= 5 or mkey:
        markers = ["o", "^", "s", "*", "+"]
    if ckey:
        dcc = dc[ckey]
        if cdict:
            dcc = dcc.map(cdict)
        normalize = Normalize(vmin=dcc.min(), vmax=dcc.max())
    for i, cond in enumerate(configs):
        idxs = [get_idx(dc, f't == {t} & {cond}') for t in range(1, 20)]
        xx = np.stack([np.mean(r['xp'][ii], axis=0) for ii in idxs])
        if ckey:
            c = np.stack([np.mean(dcc[ii], axis=0) for ii in idxs])
            c = plt.cm.get_cmap(cmap)(normalize(c))
        else:
            c = np.repeat([plt.cm.get_cmap(cmap)(
                i/len(configs))], len(xx), axis=0)

        if len(configs) <= 5:
            m = markers[i]
        elif mkey:
            m = mdict[dc[mkey][idxs[0]].iloc[0]]
        else:
            m = 'o'
        if tri:
            for (i1, d1) in enumerate(range(d-1)):
                for (i2, d2) in enumerate(range(d1+1, d)):
                    ax = axs[i2, i1]
                    sc = ax.scatter(xx[:, d1], xx[:, d2],
                                    edgecolors=c, facecolors="none",
                                    s=5, label=cond, marker=m)
                    ax.set_xlabel(f'pc{d1}')
                    ax.set_ylabel(f'pc{d2}')
        else:
            sc = ax.scatter(xx[:, 0], xx[:, 1], xx[:, 2],
                            edgecolors=c, facecolors="none", marker=m,
                            s=5, label=cond)
    handles, labels = ax.get_legend_handles_labels()
    if tri:
        ax = axs[-1, -1]
        ax.legend(handles, labels, loc='lower center')

        if evals:
            ax = figs.add_subplot(d-1, d-1, (d-1)**2-1)
            ax.set_yscale('log')
            ax.grid()
            w = 0.5
            for e in ee:
                ax.plot((-w/4, w/4), (np.abs(e), np.abs(e)),
                        c='k' if e > 0 else 'r')
            ax.set_xlim([-w/2, w/2])
            ax.get_xaxis().set_visible(False)
    else:
        ax.legend(handles, labels, loc='lower center',
                  bbox_to_anchor=(1.0, 0.5))
