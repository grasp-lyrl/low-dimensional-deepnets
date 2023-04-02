from setup import *


@call_parse
def main(
         test: Param('test', bool, default=False),
         ):


    key = 'yvh' if test else 'yh'
    fn = 'test' if test else 'train'
    if test:
        camera = dict(
            up=dict(x=0.2, y=0.2, z=-0.2),
            center=dict(x=0.1, y=-0.1, z=0),
            eye=dict(x=-0.5, y=-1.55, z=-1.5)
        )
        flip_dims = []
    else:
        camera = dict(
            up=dict(x=0.25, y=-0.1, z=1),
            center=dict(x=0, y=0.0, z=-0.4),
            eye=dict(x=1.75, y=-0.65, z=1.25)
        )
        flip_dims = []


    r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_all_geod.p")
    didx = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all_progress.p')

    emph = list(didx[didx.m == 'geodesic'].index)
    fig = plotly_3d(dc=didx.reset_index(drop=True), r=r,
                    emph={'geodesic': emph, 'p0': emph[:1], 'p*': emph[-1:]},
                    empsize={'geodesic': 4, 'p0': 8, 'p*': 8},
                    empcolor={'geodesic': 'black', 'p0': 'red', 'p*': 'red'},
                    opacity=0.25,
                    cdict=CDICT_M,
                    color_axis=False,
                    flip_dims=flip_dims,
                    ne=5,
                    size=2,
                    color='m',
                    legend=False,
                    cols=['seed', 'm', 'opt', 'err', 'verr',
                        'bs', 'aug', 'bn', 'lr', 'wd'],
                    xrange=[-1, 1],
                    yrange=[-0.5, 1.5],
                    zrange=[-0.75, 0.75]
                    )
        
    fig.update_layout(scene_camera=camera)
    fig.show()

    fig.write_image(f"../plots/all_models_{fn}_3d.pdf")
