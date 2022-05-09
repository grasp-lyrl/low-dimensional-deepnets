import argparse
from types import SimpleNamespace
import torch as th
import torch.nn.functional as F
import torchvision as thv

from utils import *
from networks import *

dev = 'cuda' if th.cuda.is_available() else 'cpu'
root = os.path.join('results', 'models', 'new')

from fastcore.script import *

def fit(m, ds, epochs=200, bs=128, autocast=True, opt=None, sched=None, fix_batch=np.zeros(2)):

    if np.max(fix_batch) == 0:
        batch_sampler = None
    else:
        batch_sampler = th.utils.data.BatchSampler(fix_batch.flatten(), bs, False)

    trainloader = th.utils.data.DataLoader(ds['train'], batch_size=bs, shuffle=True, num_workers=2, batch_sampler=batch_sampler)
    testloader = th.utils.data.DataLoader(ds['val'], batch_size=bs, shuffle=False, num_workers=2)

    def helper(t):
        m.eval()
        with th.no_grad():
            # train error
            yh, f, e = [], [], []
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(dev), y.to(dev)
                yh.append(F.log_softmax(m(x), dim=1))
                f.append(-th.gather(yh[-1], 1, y.view(-1,1)))
                e.append((y != th.argmax(yh[-1], dim=1).long()).float())

            # val error
            yvh, fv, ev = [], [], []
            for i, (xv, yv) in enumerate(testloader):
                xv, yv = xv.to(dev), yv.to(dev)
                yvh.append(F.log_softmax(m(xv), dim=1))
                fv.append(-th.gather(yvh[-1], 1, yv.view(-1,1)))
                ev.append((yv != th.argmax(yvh[-1], dim=1).long()).float())

            ss = dict(yh=yh, f=f, e=e, yvh=yvh, fv=fv, ev=ev)
            for k,_ in ss.items():
                ss[k] = th.cat(ss[k], dim=0).to('cpu')
            print('[%06d] f: %2.3f, acc: %2.2f, fv: %2.3f, accv: %2.2f, lr: %2.4f'%
                 (t, ss['f'].mean(), 100-ss['e'].mean()*100, ss['fv'].mean(), 100-ss['ev'].mean()*100, opt.param_groups[0]['lr']))
        m.train()
        return ss

    m.train()
    ss = []
    ss.append(helper(0))
    for epoch in range(epochs):
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(dev), y.to(dev)

            if not isinstance(sched, th.optim.lr_scheduler._LRScheduler):
                lr = sched(epoch + (i+1) / len(trainloader))
                opt.param_groups[0].update(lr=lr)

            with th.autocast(enabled=autocast, device_type='cuda'):
                m.zero_grad()
                yyh = m(x)
                yyh = F.log_softmax(yyh, dim=1)
                f = -th.gather(yyh, 1, y.view(-1,1)).mean()
                e = (y != th.argmax(yyh, dim=1).long()).float().mean()

            f.backward()
            opt.step()
            if isinstance(sched, th.optim.lr_scheduler._LRScheduler):
                sched.step()

        if epoch < 20:
            ss.append(helper(epoch*len(trainloader)))
        else:
            if epochs % 20 == 0 or (epoch == epochs-1):
                ss.append(helper(epoch*len(trainloader)))
    return ss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-args', '-d',  type=str,
                        default='./configs/data/cifar10-full.yaml', 
                        help='dataset name, augmentation')
    parser.add_argument('--model-args', '-m', type=str, 
                        default='./configs/model/convmixer-256-8-5-2.yaml',
                        help='model name, model aruguments, batch normalization')
    parser.add_argument('--optim-args', '-o', type=str, 
                        default='./configs/optim/adamw-500-convmixer.yaml',
                        help='batch size, batch seed, learning rate, optimizer, weight decay')
    parser.add_argument('--init-args', '-i', type=str, 
                        default='./configs/init/normal.yaml',
                        help='start from corner')

    args = parser.parse_args()

    data_args = get_configs(args.data_args)
    model_args = get_configs(args.model_args)
    optim_args = get_configs(args.optim_args)
    init_args = get_configs(args.init_args)

    args = SimpleNamespace(**{**vars(args), **data_args, **model_args, **optim_args, **init_args})
    print(args)

    fn = json.dumps(
        dict(seed=args.seed, 
            batch_seed=args.batch_seed, 
            aug=args.aug,
            m=args.m, 
            bn=args.model_args['bn'],
            opt=args.optimizer, bs=args.bs,
            lr=args.opt_args['lr'], 
            wd=args.opt_args['weight_decay'], 
            corner=args.corner)
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

    setup(args.seed)

    m = get_model(model_args, dev=dev)
    optimizer, scheduler = get_opt(optim_args, m)

    m = get_init(init_args, m)
    ss = fit(m, ds, epochs=args.epochs, bs=optim_args['bs'], autocast=optim_args['autocast'], opt=optimizer, sched=scheduler, fix_batch=fix_batch)

    th.save({"data": ss, "configs": args}, os.path.join(root, fn+'.p'))

if __name__ == '__main__':
    # default setting gives 92% acc
    main()
