import torch as th
import torch.nn.functional as F
import torchvision as thv

from utils import *

dev = 'cuda' if th.cuda.is_available() else 'cpu'
root = os.path.join('results', 'models', 'new')

from fastcore.script import *

def fit(m, ds, T=int(1e5), bs=128, autocast=True, opt=None, sched=None):

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
         model:Param('model', str, default='wr-4-8'),
         optim:Param('optimizer', str, default='sgd'),
         sched:Param('scheduler', str, default='cosine'),
         lr:Param('learning rate', float, default=0.01),
         bs:Param('batch size', int, default=200),
         momentum:Param('momentum', float, default=0.0),
         wd:Param('weight decay', float, default=0.0),
         bn:Param('batch norm', bool, default=False),
         aug:Param('data augmentation', bool, default=False),
         autocast:Param('autocast', bool, default=False)):

    args = dict(seed=seed, m=model, opt=optim, lr=lr, wd=wd, bn=bn, aug=aug, bs=bs)
    fn = json.dumps(args).replace(' ', '')
    print(fn)

    # use the same seed to setup the task
    setup(2)
    ds = get_data(dev=dev, aug=aug)

    setup(seed)

    mconfig = model.split('-')
    if mconfig[0] == 'wr':
        m = wide_resnet_t(10, int(mconfig[1]), 0, 10, int(mconfig[2]), bn=bn).to(dev)
    elif mconfig[0] == 'allcnn':
        m = allcnn_t(10, int(mconfig[1]), int(mconfig[2]), bn=bn).to(dev)
    elif mconfig[0] == 'fc':
        dims = [32*32*3] + [int(n) for n in mconfig[1:]] + [10]
        m = fcnn(dims, bn=bn).to(dev)

    # T = int(4.5e4)
    T = 180*50000//bs
    # bs = 200
    if 'sgd' in optim:
        optimizer = th.optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov='n' in optim)
    elif 'adam' in optim:
        optimizer = th.optim.Adam(m.parameters(), lr=lr, weight_decay=wd, amsgrad='ams' in optim)

    if sched == 'cosine':
        sched = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    elif sched == 'linear':
        sched = th.optim.lr_scheduler.LinearLR(optimizer)

    ss = fit(m, ds, T=T, bs=bs, autocast=autocast, opt=optimizer, sched=sched)


    th.save(ss, os.path.join(root, fn+'.p'))