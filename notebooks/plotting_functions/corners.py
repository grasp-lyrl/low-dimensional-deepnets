from setup import *


didx = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/didx_all.p')
r = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/r_yvh_all.p')

fig = plotly_3d(dc=didx.reset_index(drop=True), r=r,
                opacity=0.5,
                # cdict=CDICT_M,
                color_axis=False,
                ne=5,
                size=2,
                color='corner',
                legend=True,
                cols=['seed', 'm', 'opt', 'err', 'verr', 'corner', 't',
                      'bs', 'aug', 'lr', 'wd'],
                xrange=[r['xp'][:, 0].min(), r['xp'][:, 0].max()],
                yrange=[r['xp'][:, 1].min(), r['xp'][:, 1].max()],
                zrange=[r['xp'][:, 2].min(), r['xp'][:, 2].max()]
                )
fig.show()
df, _ = plot_explained_var(r)
print(df)
