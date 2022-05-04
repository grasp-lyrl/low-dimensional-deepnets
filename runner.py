import torch as th
import torch.nn.functional as F
import torchvision as thv

from utils import *
from networks import *

dev = 'cuda' if th.cuda.is_available() else 'cpu'
root = os.path.join('results', 'models', 'new')

from fastcore.script import *

def fit(m, ds, T=int(1e5), bs=128, autocast=True, opt=None, sched=None, fix_batch=np.zeros(2)):

    x, y = ds['x'], ds['y']
    xv, yv = ds['xv'], ds['yv']

    esteps = len(x) // bs

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
            print('[%06d] f: %2.3f, acc: %2.2f, fv: %2.3f, accv: %2.2f, lr: %2.4f'%
                 (t, ss['f'].mean(), 100-ss['e'].mean()*100, ss['fv'].mean(), 100-ss['ev'].mean()*100, opt.param_groups[0]['lr']))
        m.train()
        return ss

    m.train()
    iis = np.arange(x.shape[0])
    ss = []
    ss.append(helper(0))
    for t in tqdm.tqdm(range(1, T+1)):
        if np.max(fix_batch) == 0:
            # ii = np.random.choice(iis, bs)
            bi = t % esteps
            if bi == 0:
                iis = iis[np.random.permutation(x.shape[0])]
            ii = iis[bi*bs:(bi+1)*bs]
        else:
            ii = fix_batch[t]
        xx, yy = x[ii], y[ii]

        lr = sched(t / float(esteps))
        opt.param_groups[0].update(lr=lr)
        with th.autocast(enabled=autocast, device_type='cuda'):
            m.zero_grad()
            yyh = m(xx)
            yyh = F.log_softmax(yyh, dim=1)
            f = -th.gather(yyh, 1, yy.view(-1,1)).mean()
            e = (yy != th.argmax(yyh, dim=1).long()).float().mean()

        f.backward()
        opt.step()
        # sched.step()

        if t < esteps*25:
            if t % esteps == 0:
                ss.append(helper(t))
        else:
            if t %(20*esteps) == 0 or (t == T):
                ss.append(helper(t))
    return ss

@call_parse
def main(seed:Param('seed', int, default=42),
         model:Param('model', str, default='wr-4-8'),
         optim:Param('optimizer', str, default='sgd'),
         sched:Param('scheduler', str, default='cosine'),
         lr:Param('learning rate', float, default=0.01),
         bs:Param('batch size', int, default=200),
         epochs:Param('epochs', int, default=200),
         momentum:Param('momentum', float, default=0.0),
         wd:Param('weight decay', float, default=0.0),
         bn:Param('batch norm', bool, default=False),
         aug:Param('data augmentation', str, default='none'),
         corner:Param('corner', str, default='normal', choices=['normal','uniform','subsample-200', 'subsample-2000']),
         batch_seed:Param('batch_seed', int, default=-1),
         autocast:Param('autocast', bool, default=False)):

    args = dict(seed=seed, batch_seed=batch_seed, m=model, opt=optim, lr=lr, wd=wd, bn=bn, aug=aug, 
                bs=bs, corner=corner)
    fn = json.dumps(args).replace(' ', '')
    print(fn)
    # save the configuaration to ss check fname does not overflow.

    # use the same seed to setup the task
    setup(2)
    ds = get_data(dev=dev, aug=aug)

    if batch_seed < 0:
        fix_batch = np.zeros(2)
    else:
        setup(batch_seed)
        fix_batch = np.random.randint(ds['x'].shape[0], size=(T, bs))

    setup(seed)

    mconfig = model.split('-')
    if mconfig[0] == 'wr':
        depth, widen_factor, in_planes = mconfig[1:] 
        m = wide_resnet_t(int(depth), int(widen_factor), dropout_rate=0, num_classes=10, in_planes=int(in_planes), bn=bn)
    elif mconfig[0] == 'allcnn':
        c1, c2 = mconfig[1:]
        m = allcnn_t(10, int(c1), int(c2), bn=bn).to(dev)
    elif mconfig[0] == 'fc':
        dims = [32*32*3] + [int(n) for n in mconfig[1:]] + [10]
        m = fcnn(dims, bn=bn).to(dev)
    elif mconfig[0] == 'convmixer':
        dim, depth = mconfig[1:]
        m = convmixer(int(dim), int(depth), kernel_size=5, patch_size=2, n_classes=10).to(dev)

    T = epochs*50000//bs
    if 'sgd' in optim:
        optimizer = th.optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov='n' in optim)
    elif 'adam' in optim:
        optimizer = th.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

    if sched == 'cosine':
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    elif sched == 'linear':
        scheduler = th.optim.lr_scheduler.LinearLR(optimizer)
    elif sched == 'cosine_with_warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=T//2, cycle_mult=1,
                                            max_lr=lr, warmup_steps=10, gamma=0.1)

    elif sched == 'convmixer':
        scheduler = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs],
                                     [0, lr, lr/20.0, 0])[0]

    if corner == 'uniform':
        if bn==False:
            print("cannot train to the corner, use bn=True")
            return
        opt_init = th.optim.SGD(m.parameters(), lr=0.05, momentum=momentum, weight_decay=wd, nesterov='n' in optim)
        ds_init = relabel_data(ds, frac=1)
        ss_init = fit(m, ds_init, T=T, bs=bs, autocast=autocast, opt=opt_init, sched = sched)
    elif corner.split('-')[0] == 'subsample':
        ds_init = get_data(dev=dev, aug=aug, sub_sample=int(corner.split('-')[1]), shuffle=True)
        ss_init = fit(m, ds_init, T=T//10, bs=bs, autocast=autocast, opt=optimizer, sched = sched)


    ss = fit(m, ds, T=T, bs=bs, autocast=autocast, opt=optimizer, sched=scheduler, fix_batch=fix_batch)

    th.save(ss, os.path.join(root, fn+'.p'))
    # update to s3 and delete local file
