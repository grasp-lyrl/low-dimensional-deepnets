from setup import *

@call_parse
def main(all: Param('all', bool, default=False), 
    main: Param('main', bool, default=False), 
    ps: Param('ps', bool, default=False), 
    err: Param('err', bool, default=False), 
    spread: Param('spread', bool, default=False),
    test: Param('test', bool, default=False),
    ): 

    key = 'yvh' if test else 'yh'
    fn = 'test' if test else 'train'
    all_centers = [0, 0.3, 0.1]
    flip_dims = [0, 2] if test else None
    radius = 0.3 if test else 0.15

    r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all_geod.p")
    didx = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all_progress.p')
    emph = list(didx[didx.m == 'geodesic'].index)

    # PC 1-3, colored by model architecture
    if main or all:
        centers = all_centers
        centers= [0, 0.4, 0.1]
        grid_ratio=[4, 3, 2]
        grid_size=0.2
        f, gs = triplot(didx, r, 
                        emph={'geodesic': emph, 'p0': emph[:1], 'p*': emph[-1:]},
                        empsize={'geodesic': 6, 'p0': 55, 'p*': 55, 'ref': 8}, 
                        empcolor={'geodesic': 'black', 'p0': 'red',
                                'p*': 'red', 'ref': 'red'},
                        grid_ratio=grid_ratio, grid_size=grid_size, centers=centers,
                        flip_dims=flip_dims,
                        ax_label=True,
                        ckey='m', cdict=CDICT_M,
                        legend=False,
                        )
        f.savefig(f'../plots/all_models_{fn}_2d.pdf', bbox_inches='tight')

    # PC 1&2, include all models, colored by loss
    if ps or all:
        centers=[0.7,2]
        grid_ratio=[3, 3]
        grid_size=0.8
        ckey = 'vfavg' if test else 'favg'
        f, gs = triplot(didx, r, 
                        emph={'geodesic':emph, 'p0':emph[:1], 'p*':emph[-1:]},
                            empsize={'geodesic':6, 'p0':55, 'p*':55, 'ref':8}, 
                            empcolor={'geodesic':'black', 'p0':'red', 'p*':'red', 'ref':'red'}, 
                        cmin=0, cmax=3,
                        ckey=ckey, 
                        legend=True, cbar_title=f'{fn} loss',
                        grid_ratio=grid_ratio, grid_size=grid_size, centers=centers,
                        flip_dims=flip_dims,
                    )
        f.savefig(f'../plots/all_models_{fn}_2d_ps.pdf', bbox_inches='tight')

    # PC 1&2, colored by error 
    if err or all:
        centers=all_centers[:2]
        grid_ratio=[3, 3]
        grid_size=0.3
        ckey = 'verr' if test else 'err'
        f, gs = triplot(didx, r, 
                        emph={'geodesic':emph, 'p0':emph[:1], 'p*':emph[-1:]},
                        empsize={'geodesic':6, 'p0':55, 'p*':55, 'ref':8}, 
                        empcolor={'geodesic':'black', 'p0':'red', 'p*':'red', 'ref':'red'}, 
                        cmin=0, cmax=1,
                        ckey=ckey,
                        discrete_c=True, cbins=10,
                        legend=True, cbar_title=f'{fn} error',
                        grid_ratio=grid_ratio, grid_size=grid_size, centers=centers,
                        flip_dims=flip_dims,
                    )
        f.savefig(f'../plots/all_models_{fn}_2d_error.pdf', bbox_inches='tight')

    # PC 1&2, colored by distance to three reference points 
    if spread or all:
        refs = [[151505], [151459], [151409]] # 0.99, 0.51, 0.01 progress on geodesic
        d2ref = []
        with h5py.File(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/w_{key}_all_geod.h5", "r") as f:
            w = f["w"]
            for ri in refs:
                d2ref.append(w[ri, :])
        d2ref = np.stack(d2ref).squeeze()
        didx['label'] = np.argmin(d2ref, axis=0).astype(str)
        didx.loc[np.where(d2ref.min(axis=0) > radius)[0], 'label'] = 'none'

        centers=all_centers[:2]
        grid_ratio=[3, 3]
        grid_size=0.3
        cdict = {l: plt.get_cmap('rocket', 4).colors[i] for (i,l) in enumerate(['none', '2', '1', '0'])}
        f, gs = triplot(didx, r, 
                        emph={
                            'geodesic':emph,
                            'ref1': refs[0], 'ref2': refs[1], 'ref3':refs[2],
                        },
                        empsize={
                            'geodesic':6,  
                            'ref1':55, 'ref2':55 , 'ref3':55
                        }, 
                        empcolor={
                            'geodesic':'black', 
                            'ref1':'black', 'ref2':'black', 'ref3':'black',
                        }, 
                        ckey='label', cdict=cdict, 
                        legend=True, cbar_title='label',
                        grid_ratio=grid_ratio, grid_size=grid_size, centers=centers,
                        flip_dims=flip_dims,
                    )
        f.savefig(f'../plots/all_models_{fn}_2d_spread.pdf', bbox_inches='tight')