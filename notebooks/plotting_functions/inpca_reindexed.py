from setup import *


@call_parse
def main(
    test: Param('test', bool, default=False),
    mode: Param('mode', str, default='mean'),
    t: Param('t', int, default=100),
    weight_geod: Param('weight_geod', bool, default=False),
         ):

    key = 'yvh' if test else 'yh'

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

    if mode == 'mean':
        w = dists[:t, :, :].sum(0)/100
    else:
        w = dists[t, :, :]

    weight = np.ones(len(w))
    if weight_geod:
        iw = get_idx(didx, "m=='geodesic'")
        weight[iw] = len(didx)//10
    weight /= weight.sum()

    dmean = (w*weight).sum(1, keepdims=True)
    w = w - dmean
    w = w - (w*weight[:, None]).sum(0)
    w = -0.5*w
    r = proj_(w, w.shape[0], 3)

    emph = list(didx[didx.m == 'geodesic'].index)
    f, gs = triplot(didx.reset_index(drop=True), r, emph={'geodesic': emph},
                    empsize={'geodesic': 12}, cdict=CDICT_M,
                    empcolor={'geodesic': 'black'}, d=4,
                    ckey='m', colorscale='Set1',
                    grid_ratio=[4, 4], grid_size=0.06, centers=[0, 0])
    fn = 'test' if test else 'train'
    f.savefig(f'../plots/{fn}_trajectories_inpca.pdf')
