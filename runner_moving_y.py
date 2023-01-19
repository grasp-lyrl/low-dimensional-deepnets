from fastcore.script import *
import argparse
from types import SimpleNamespace
import torch as th
import torch.nn.functional as F
import numpy as np
import time
import os
import json

from utils import get_configs, get_data, get_model, get_opt, get_init, setup
from reparameterization import gamma

dev = 'cuda' if th.cuda.is_available() else 'cpu'


def fit(m, ds, epochs=200, bs=128, autocast=True, opt=None, sched=None, 
        fix_batch=np.zeros(2), fname='', progress_fn=lambda x: x):

    if np.max(fix_batch) == 0:
        batch_sampler = None
    else:
        batch_sampler = th.utils.data.BatchSampler(
            fix_batch.flatten(), bs, False)

    trainloader = th.utils.data.DataLoader(
        ds['train'], batch_size=bs, shuffle=True, num_workers=0, batch_sampler=batch_sampler)
    fixed_trainloader = th.utils.data.DataLoader(
        ds['train'], batch_size=bs, shuffle=False, num_workers=0)
    testloader = th.utils.data.DataLoader(
        ds['val'], batch_size=bs, shuffle=False, num_workers=0)

    def helper(t):
        m.eval()
        start = time.time()
        with th.no_grad():
            # train error
            yh, f, e = [], [], []
            for i, (x, y) in enumerate(fixed_trainloader):
                x, y = x.to(dev), y.to(dev)
                yh.append(F.log_softmax(m(x), dim=1))
                f.append(-th.gather(yh[-1], 1, y.view(-1, 1)))
                e.append((y != th.argmax(yh[-1], dim=1).long()).float())

            # val error
            yvh, fv, ev = [], [], []
            for i, (xv, yv) in enumerate(testloader):
                xv, yv = xv.to(dev), yv.to(dev)
                yvh.append(F.log_softmax(m(xv), dim=1))
                fv.append(-th.gather(yvh[-1], 1, yv.view(-1, 1)))
                ev.append((yv != th.argmax(yvh[-1], dim=1).long()).float())

            ss = dict(yh=yh, f=f, e=e, yvh=yvh, fv=fv, ev=ev)
            for k, _ in ss.items():
                ss[k] = th.cat(ss[k], dim=0).to('cpu')
            f = ss['f'].mean()
            acc = 100-ss['e'].mean()*100
            fv = ss['fv'].mean()
            accv = 100-ss['ev'].mean()*100
            lr = opt.param_groups[0]['lr']
            print('[%06d] f: %2.3f, acc: %2.2f, fv: %2.3f, accv: %2.2f, lr: %2.4f, time: %2.2f' %
                  (t, f, acc, fv, accv, lr, time.time()-start))

        m.train()
        return ss

    scaler = th.cuda.amp.GradScaler()
    m.train()
    ss = []
    t = 0
    ss.append(helper(t))
    for epoch in range(epochs):
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(dev), y.to(dev)
            y_ = np.array(y.cpu().numpy(), dtype=np.int32)
            yt = np.zeros((y_.size, y_.max() + 1))
            yt[np.arange(y_.size), y_] = 1
            ti = progress_fn(t)
            yt = th.Tensor(gamma(ti, np.ones_like(yt) / yt.shape[1], yt) ** 2).to(dev)

            with th.autocast(enabled=autocast, device_type='cuda'):
                m.zero_grad()
                yyh = m(x)
                yyh = F.log_softmax(yyh, dim=1)
                f = -(yyh * yt).sum(1).mean()
                e = (y != th.argmax(yyh, dim=1).long()).float().mean()

            if autocast:
                scaler.scale(f).backward()
                scaler.step(opt)
                scaler.update()
            else:
                f.backward()
                opt.step()
            sched.step()
            t += 1
            if epoch < 5 and i % (len(trainloader) // 4) == 0:
                ss.append(helper(t))

        if 5 <= epoch <= 25:
            ss.append(helper(t))
        elif 25 < epoch <= 65 and epoch % 4 == 0:
            ss.append(helper(t))
        elif epoch > 65 and epoch % 15 == 0 or (epoch == epochs-1):
            ss.append(helper(t))
    return ss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-config', '-d',  type=str,
                        default='./configs/data/cifar10-full.yaml',
                        help='dataset name, augmentation')
    parser.add_argument('--model-config', '-m', type=str,
                        default='./configs/model/convmixer.yaml',
                        help='model name, model aruguments, batch normalization')
    parser.add_argument('--optim-config', '-o', type=str,
                        default='./configs/optim/adam-200-0.001.yaml',
                        help='batch size, batch seed, learning rate, optimizer, weight decay')
    parser.add_argument('--init-config', '-i', type=str,
                        default='./configs/init/normal.yaml',
                        help='start from corner')
    parser.add_argument('--save-dir', '-s', type=str, 
                        default='results/models/all',
                        help='directory to save results')

    args = parser.parse_args()
    seed = args.seed

    root = args.save_dir
    data_args = get_configs(args.data_config)
    model_args = get_configs(args.model_config)
    optim_args = get_configs(args.optim_config)
    init_args = get_configs(args.init_config)

    args = SimpleNamespace(
        **{**vars(args), **data_args, **model_args, **optim_args, **init_args})
    print(args)

    fn = json.dumps(
        dict(seed=seed,
             bseed=args.batch_seed,
             aug=args.aug,
             m=os.path.basename(args.model_config)[:-5],
             bn=args.model_args['bn'],
             drop=args.model_args['dropout_rate'],
             opt=os.path.basename(args.optim_config).split('-')[0],
             bs=args.bs,
             lr=args.opt_args['lr'],
             wd=args.opt_args['weight_decay'])
    ).replace(' ', '')
    print(fn)

    setup(2)
    ds = get_data(data_args)
    N_train = len(ds['train'])
    T = args.epochs * N_train // args.bs
    optim_args['T'] = T

    if args.batch_seed < 0:
        fix_batch = np.zeros(2)
    else:
        setup(args.batch_seed)
        fix_batch = np.random.randint(N_train, size=(T, args.bs))

    setup(seed)

    m = get_model(model_args, dev=dev)
    optimizer, scheduler = get_opt(optim_args, m)

    m = get_init(init_args, m)

    progress_fn = lambda t: max((t + 10*(N_train // args.bs)), T) / T
    ss = fit(m, ds, epochs=args.epochs, bs=optim_args['bs'], autocast=optim_args['autocast'],
             opt=optimizer, sched=scheduler, fix_batch=fix_batch, fname=fn, progress_fn=progress_fn)

    th.save({"data": ss, "configs": args}, os.path.join(root, fn+'.p'))


if __name__ == '__main__':
    # default setting gives 92% acc
    main()
