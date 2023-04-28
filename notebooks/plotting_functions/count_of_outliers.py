from setup import *

key = 'yh'
fn = 'all_geod'
r = th.load(f"/home/ubuntu/ext_vol/inpca/inpca_results_all/r_{key}_{fn}.p")
didx = th.load(
    '/home/ubuntu/ext_vol/inpca/inpca_results_all/didx_geod_all_progress.p').reset_index(drop=True)

# d_out = didx[didx.favg >= 5].reset_index()
d_out = didx[didx.d2truth >= 2].reset_index()

# outliers by nepochs
bins = np.array([0, 5, 10, 15, 25, 50, 75, 100, 150])
ind = np.digitize(d_out['nepochs'], bins)
d_out['nepochs_binned'] = bins[ind-1]

f = plt.figure(figsize=(25, 10))
ax = sns.countplot(data=d_out, x='nepochs_binned', hue='opt')
ax.legend(loc='upper right')
ax.set(ylabel='# of models with $d_B(P, P_*)>2$')
ax.set(xlabel='Epochs')
new_ticks = [i.get_text() for i in ax.get_xticklabels()]
new_ticks = ['-'.join(new_ticks[i:i+2])
             for i in range(len(new_ticks)-1)] + ['>150']
ticks = ax.set_xticks(np.arange(0, len(new_ticks)), new_ticks)
f.savefig('../plots/all_models_train_by_nepochs.pdf')

# outliers by model architecture
f = plt.figure(figsize=(25, 10))
ax = sns.countplot(data=d_out, x='m', hue='opt')
ax.legend(loc='upper right')
ax.set(ylabel='# of models with $d_B(P, P_*)>2$')
ax.set(xlabel='Model Architecture')
f.savefig('../plots/all_models_train_by_m.pdf')
