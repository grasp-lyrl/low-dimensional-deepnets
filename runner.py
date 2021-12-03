import torch as th
import torch.nn.functional as F
import torchvision as thv

from utils import *

dev = 'cuda' if th.cuda.is_available() else 'cpu'
root = os.path.join('results', 'models')

from fastcore.script import *

def fit(m, ds, T=int(1e5), bs=128, autocast=True):
    opt = th.optim.SGD(m.parameters(), lr=0.05)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T)

    x, y = ds['x'], ds['y']
    xv, yv = ds['xv'], ds['yv']

    def helper(t):
        m.eval()
        with th.no_grad():
            # train error
            yh, f, e = [], [], []
            for ii in th.chunk(th.arange(x.shape[0]), x.shape[0]//bs):
                xx, yy = x[ii], y[ii]            
                yh.append(F.log_softmax(m(xx), dim=1))
                f.append(-th.gather(yh[-1], 1, yy.view(-1,1)))
                e.append((yy != th.argmax(yh[-1], dim=1).long()).float())

            # val error
            yvh, fv, ev = [], [], []
            for ii in th.chunk(th.arange(xv.shape[0]), xv.shape[0]//bs):
                xxv, yyv = xv[ii], yv[ii]
                yvh.append(F.log_softmax(m(xxv), dim=1))
                fv.append(-th.gather(yvh[-1], 1, yyv.view(-1,1)))
                ev.append((yyv != th.argmax(yvh[-1], dim=1).long()).float())

            ss = dict(yh=yh, f=f, e=e, yvh=yvh, fv=fv, ev=ev)
            for k,_ in ss.items():
                ss[k] = th.cat(ss[k], dim=0).to('cpu')
            print('[%06d] f: %2.3f, e: %2.2f, fv: %2.3f, ev: %2.2f'%
                 (t, ss['f'].mean(), ss['e'].mean()*100, ss['fv'].mean(), ss['ev'].mean()*100))
        m.train()
        return ss

    m.train()
    iis = np.arange(x.shape[0])
    ss = []
    for t in tqdm.tqdm(range(T)):
        ii = np.random.choice(iis, bs)
        xx, yy = x[ii], y[ii]

        with th.autocast(enabled=autocast, device_type='cuda'):
            m.zero_grad()
            yyh = m(xx)
            yyh = F.log_softmax(yyh, dim=1)
            f = -th.gather(yyh, 1, yy.view(-1,1)).mean()
            e = (yy != th.argmax(yyh, dim=1).long()).float().mean()

        f.backward()
        opt.step()
        sched.step()

        if t < T//10:
            if t % (T//100) == 0:
                ss.append(helper(t))
        else:
            if t%(T//10) == 0 or (t == T-1):
                ss.append(helper(t))
    return ss

@call_parse
def main(seed:Param('seed', int, default=42),
         widen:Param('widen', int, default=2),
         numc:Param('numc', int, default=2),
         noise_label:Param('noise_label', bool, default=False),
         relabel:Param('relabel', float, default=0),
         autocast:Param('autocast', bool, default=False)):

    args = dict(seed=seed, widen=widen, numc=numc, noise_label=noise_label)

    # use the same seed to setup the task
    setup(2)
    ds = get_data(sub_sample=0.5, dev=dev)
    if noise_label:
        print('Using noisy labels')
        ds['y'] = ds['y'][th.randperm(len(ds['y']))]
    if relabel > 0:
        args.update(dict(relabel=0.0))
        fn = json.dumps(args).replace(' ', '') + '.p'
        print(f"Relabeling {relabel*100:.2f}% of data, using model {fn}")
        relabel_data(os.path.join(root, fn), ds['y'], relabel)

    setup(seed)
    m = wide_resnet_t(10, widen, 0, 10, numc).to(dev)
    ss = fit(m, ds, T=int(4e4), bs=128, autocast=autocast)

    args.update(dict(relabel=relabel))
    fn = json.dumps(args).replace(' ', '')
    th.save(ss, os.path.join(root, fn+'.p'))