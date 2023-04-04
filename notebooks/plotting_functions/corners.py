from setup import *


@call_parse
def main(
      k: Param('k', str, default='yh')
):
      didx = th.load('/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/didx_test.p')
      r = th.load(f'/home/ubuntu/ext_vol/inpca/inpca_results_all/corners/r_{k}_test.p')

      fig = plotly_3d(dc=didx.reset_index(drop=True), r=r,
                  opacity=0.5,
                  color_axis=False,
                  # cdict=CDICT_M,
                  ne=5,
                  size=2,
                  color='iseed',
                  colorscale='Set1',
                  legend=True,
                  cols=['seed', 'iseed', 'm', 'opt', 'err', 'verr', 'corner', 't',
                        'bs', 'aug', 'lr', 'wd'],
                  xrange=[r['xp'][:, 0].min(), r['xp'][:, 0].max()],
                  yrange=[r['xp'][:, 1].min(), r['xp'][:, 1].max()],
                  zrange=[r['xp'][:, 2].min(), r['xp'][:, 2].max()]
                  )
      fig.show()
      df, _ = plot_explained_var(r, key=k)
      print(df)
