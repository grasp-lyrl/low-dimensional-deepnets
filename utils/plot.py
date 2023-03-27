import torch.nn.functional as F
import torch as th
from bokeh import palettes
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from utils import CDICT_M


def triplot(dc, r, d=3, emph=[], ckey='', empcolor={}, empsize={},
            cdict=None, cmin=None, cmax=None, discrete_c=False, cbins=None, colorscale='vlag',
            grid_size=0.25, grid_ratio=[5, 3, 2], centers=[0, 0, 0],
            wspace=0, hspace=0,
            ax_label=False, legend=False):

    widths = grid_ratio[1:]
    heights = grid_ratio[:-1]
    d = len(grid_ratio)
    ncols, nrows = d-1, d-1
    if legend:
        widths.append(0.1)
        ncols += 1
    fig = plt.figure(figsize=(8*sum(widths)/sum(heights), 8))
    gs = GridSpec(nrows, ncols, width_ratios=widths, height_ratios=heights)

    if len(emph) > 0:
        iie = []
        for (_, ie) in emph.items():
            iie.extend(ie)
        d_ = dc.drop(iie)
        xx = r['xp'][d_.index, :]
        d_ = d_.reset_index(drop=True)
    else:
        d_ = dc
        xx = r['xp']

    ee = r['e']
    s = 2

    c = d_[ckey]
    if cdict is not None:
        c = c.map(cdict)
    else:
        if discrete_c:
            if cbins is None:
                cdict = {c: i for (i, c) in enumerate(c.unique())}
                # print(cdict)
                c = c.map(cdict)
                cbins = len(c.unique())
            colorscale = plt.get_cmap(colorscale, cbins)
    cmin = cmin or c.min()
    cmax = cmax or c.max()

    for d1 in range(d-1):
        for d2 in range(d1+1, d):

            ax = fig.add_subplot(gs[d1, d2-1])
            sc = ax.scatter(xx[:, d2], xx[:, d1],
                            c=c, vmin=cmin, vmax=cmax, 
                            s=s, lw=0.5, alpha=0.5, cmap=colorscale,
                            rasterized=True)
            if len(emph) > 0:
                for (name, ie) in emph.items():
                    sc_emph = ax.scatter(r['xp'][ie, d2], r['xp'][ie, d1], c=empcolor.get(name, 'darkred'),
                                         rasterized=True,
                                         s=empsize.get(name, 8), lw=0.5, alpha=0.5)
            ax.set_xlim([centers[d2]-grid_size*grid_ratio[d2],
                        centers[d2]+grid_size*grid_ratio[d2]])
            ax.set_ylim([centers[d1]-grid_size*grid_ratio[d1],
                        centers[d1]+grid_size*grid_ratio[d1]])

            if ee[d2] < 0:
                ax.spines['bottom'].set_color('red')
            if ee[d1] < 0:
                ax.spines['left'].set_color('red')

            if d2 == d1+1:
                if ax_label:
                    ax.set_xlabel(f'PC{d2+1}')
                    ax.set_ylabel(f'PC{d1+1}')
                ax.grid(False)
                ax.set_xticks([0])
                if d1 == 0:
                    ax.set_yticks([0, centers[d1] + grid_size*grid_ratio[d1]])
                else:
                    ax.set_yticks([0])
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    if legend:
        ax = fig.add_subplot(gs[:, -1])
        if cdict is not None:
            ticks = list(cdict.keys())
            boundaries = [cdict[k] for k in ticks]
            clb = plt.colorbar(sc, cax=ax, boundaries=boundaries, ticks=boundaries)
            clb.ax.set_yticklabels(ticks)
        else:
            clb = plt.colorbar(
                sc, cax=ax)
        clb.ax.set_title(ckey)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.show()
    return fig, gs

def plotly_3d(dc, r, emph=[], empcolor={}, empsize={}, empmode='markers',
              return_d=False, ne=3, dims=[1, 2, 3],
              size=5,
              cdict=None,
              cols=['seed', 'm', 'opt', 'err', 'verr',
                    'bs', 'aug', 'bn', 'lr', 'wd'],
              color='t', colorscale='RdBu', mode='markers',
              xrange=[-1, 1], yrange=[-1, 1], zrange=[-1, 1], opacity=0.7):

    for i in range(ne):
        dc[f"x{i+1}"] = r['xp'][:, i]

    d_ = pd.DataFrame()

    for col in cols:
        d_[col] = f'{col}: ' + dc[col].astype(str)
    text = d_[cols].apply("<br>".join, axis=1)
    if 't' in cols:
        text = 't: ' + dc['t'].astype(str) + '<br>' + text

    c = dc[color]
    discrete_c = len(c.unique()) < 10
    if discrete_c and cdict is None:
        colors = getattr(palettes, colorscale)[max(len(c.unique()), 3)]
        cdict = {c: colors[i] for (i, c) in enumerate(c.unique())}
        print(cdict)

    fig = go.Figure()

    if len(emph) > 0:
        iie = []
        for (_, ie) in emph.items():
            iie.extend(ie)
        d_ = dc.drop(iie).reset_index(drop=True)
    else:
        d_ = dc

    if discrete_c:

        fig.add_trace(
            go.Scatter3d(
                x=d_[f'x{dims[0]}'],
                y=d_[f'x{dims[1]}'],
                z=d_[f'x{dims[2]}'],
                marker=dict(
                    size=size,
                    opacity=opacity,
                    color=c.map(cdict),
                ),
                hovertemplate='<b>%{text}</b><extra></extra>',
                text=text,
                mode='markers',
                showlegend=True,
                name='main',
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=d_[f'x{dims[0]}'],
                y=d_[f'x{dims[1]}'],
                z=d_[f'x{dims[2]}'],
                marker=dict(
                    size=size,
                    opacity=opacity,
                    color=c,
                    colorscale=colorscale
                ),
                hovertemplate='<b>%{text}</b><extra></extra>',
                text=text,
                mode='markers',
                showlegend=True,
                name='main'
            )
        )

    if len(emph) > 0:
        for (name, ie) in emph.items():
            demph = dc.iloc[ie]
            iie.extend(ie)
            fig.add_trace(
                go.Scatter3d(
                    x=demph[f'x{dims[0]}'],
                    y=demph[f'x{dims[1]}'],
                    z=demph[f'x{dims[2]}'],
                    marker=dict(
                        size=empsize.get(name, 4),
                        opacity=0.9,
                        color=empcolor.get(name, 'darkred'),
                    ),
                    mode=empmode,
                    name=name,
                    showlegend=True
                )
            )
    axis_color = ['red' if r['e'][i] < 0 else 'black' for i in range(3)]
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        scene_xaxis_range=xrange,
        scene_yaxis_range=yrange,
        scene_zaxis_range=zrange,
        template='plotly_white',
        scene_xaxis_linecolor=axis_color[0],
        scene_xaxis_linewidth=4.5,
        scene_yaxis_linecolor=axis_color[1],
        scene_yaxis_linewidth=4.5,
        scene_zaxis_linecolor=axis_color[2],
        scene_zaxis_linewidth=4.5,
        scene_aspectratio={
            'x': xrange[1]-xrange[0],
            'y': yrange[1]-yrange[0],
            'z': zrange[1]-zrange[0],
        },
        legend=dict(itemsizing="constant",
                    itemwidth=100,
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01)
    )
    print(r['e'])

    return fig


def plot_pairwise_dist(dists, configs,
                       cconds=[lambda x: True],
                       rconds=[lambda x: True],
                       tidxs=slice(0, 100),
                       sortby=1, reduction='avg_pool2d', annot=False,
                       xblock_size=10, yblock_size=1,
                       label_idxs=slice(0, -1),
                       square=False, return_didx=False):
    cidxs = []
    ridxs = []
    columns = []
    rows = []
    if square:
        rconds = cconds
        yblock_size = xblock_size
    if cconds or rconds:
        for (i, c) in configs.iterrows():
            c = list(c)
            if cconds and all(f(c) for f in cconds):
                cidxs.append(i)
                columns.append(c)
            if rconds and all(f(c) for f in rconds):
                ridxs.append(i)
                rows.append(c)
    else:
        cidxs, ridxs = range(len(configs)), range(len(configs))
        columns, rows = configs, configs
    cidxs, ridxs = np.array(cidxs), np.array(ridxs)
    columns, rows = np.stack(columns), np.stack(rows)
    clidxs = (np.lexsort(
        np.stack([columns[:, sidx:sidx+1].T for sidx in sortby]))).squeeze()
    cidxs = cidxs[clidxs]
    columns = columns[clidxs, label_idxs]
    rlidxs = (np.lexsort(
        np.stack([rows[:, sidx:sidx+1].T for sidx in sortby]))).squeeze()
    ridxs = ridxs[rlidxs]
    rows = rows[rlidxs, label_idxs]

    didxs = dists[tidxs, cidxs, :][:, :, ridxs].squeeze()
    if reduction:
        didxs = getattr(F, reduction)(th.Tensor(didxs).unsqueeze(0).unsqueeze(0),
                                      kernel_size=(yblock_size, xblock_size),
                                      stride=(yblock_size, xblock_size)).squeeze().numpy()
        columns = columns[::xblock_size]
        rows = rows[::yblock_size]
    if return_didx:
        return didxs, rows, columns

    if reduction:
        ax = sns.heatmap(pd.DataFrame(didxs, columns=columns,
                         index=rows), annot=annot, fmt='.2g')
    else:
        ax = sns.heatmap(pd.DataFrame(didxs, columns=columns, index=rows),
                         xticklabels=xblock_size, yticklabels=yblock_size, annot=annot, fmt='.2g')
    return ax, didxs, rows, columns


def plot_dendrogram(linkage, ylabels, cdict, didx, color_by=0,
                    color_threshold=0.5,
                    above_threshold_color='C0',
                    cols=['m', 'opt', 'bs', 'lr', 'wd', 'aug'], 
                    key='yh',
                    ):
    idxs = didx.groupby(cols).indices

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios':[3, 1]}, figsize=(10, 24))

    dend = sch.dendrogram(linkage, orientation='right',
                        no_plot=True, color_threshold=color_threshold)
    label_colors = [cdict[m] for m in ylabels[dend['leaves'], color_by]]
    dend = sch.dendrogram(linkage, orientation='right', labels=np.array(
        ylabels[:, 0]), color_threshold=color_threshold, ax=ax[0],
        above_threshold_color=above_threshold_color)

    # barplot of errors
    vkey = 'verr' if key == 'yvh' else 'err'
    err = []
    for c in ylabels[dend['leaves'][::-1], :]:
        err.append(didx.iloc[idxs[tuple(c)]][vkey].mean())

    ylabels_combined = ['.'.join(l) for l in ylabels.astype(str)]
    models = np.array(ylabels_combined)[dend['leaves'][::-1]]
    data = pd.DataFrame(np.vstack([models, err]).T, columns=['model', 'err'])
    data['err'] = data['err'].astype(float)

    barc = [cdict[m.split('.')[color_by]] for m in models]
    sns.barplot(x="err", y="model", data=data, ax=ax[1], palette=barc).set(
        yticklabels=[], ylabel="")
    ax[1].set_xlim([0, 1])
    ax[1].invert_xaxis()
    ax[1].yaxis.set_ticks_position('none')
    ax[1].grid(axis='y')


    yax = ax[0].get_yaxis()
    r = yax.set_tick_params(pad=103)
    for (ik, k) in enumerate([1, 2, 3, 4, 5]):
        secax = ax[0].secondary_yaxis('left')
        secax.set_yticks(ax[0].get_yticks())
        secax.set_yticklabels(ylabels[dend['leaves'], k])
        secax.get_yaxis().set_tick_params(pad=103-(ik+1)*20, labelsize=5, length=0)
        for (n, y) in enumerate(secax.get_ymajorticklabels()):
            y.set_color(label_colors[n])

    plt.subplots_adjust(wspace=0)
    for (n, x) in enumerate(ax[0].get_ymajorticklabels()):
        # x.set_color(cdict[x.get_text().split(' ')[color_by]])
        x.set_color(label_colors[n])
    return fig, dend 

def plot_evals(r):
    fig, ax = plt.subplots(1, figsize=(6, 10))
    ax.set_yscale('log')
    ax.grid()
    w = 0.5
    for e in r['es']:
        ax.plot((-w/4, w/4), (np.abs(e), np.abs(e)),
                c='k' if e > 0 else 'r')
    ax.set_xlim([-w/3, w/3])
    ax.get_xaxis().set_visible(False)
    return fig

def plot_cone(p):
    # P=[p1,p2,p3], coordinates of the tip of cone
    u = np.linspace(0, 5, 240)
    v = np.linspace(0, 2 * np.pi, 240)
    uGrid, vGrid = np.meshgrid(u, v)

    x = uGrid * np.cos(vGrid) + p[0]
    y = uGrid + p[1]
    z = uGrid * np.sin(vGrid) + p[2]


def plot_explained_var(r, key='yh'):
    ii = np.argsort(np.abs(r['es']))[::-1]
    es = r['es'][ii][:50]
    dset = 'train' if key == 'yh' else 'test'
    df = pd.DataFrame({'eigenvalue index':np.arange(len(es)), 'explained variance':1 - np.sqrt(1-np.cumsum(es ** 2)/r['fn']**2), 
                       'data':dset})
    f = plt.figure(figsize=(8, 6))
    g = sns.lineplot(data=df, x='eigenvalue index',
                 y='explained variance', hue='data', marker="o")
    return df, f
