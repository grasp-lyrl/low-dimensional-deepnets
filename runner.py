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
            ii = np.random.choice(iis, bs)
            # bi = (t-1) % esteps
            # if bi == 0:
            #     iis = iis[np.random.permutation(x.shape[0])]
            # ii = iis[bi*bs:(bi+1)*bs]
        else:
            ii = fix_batch[t]
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

        if t < esteps*25:
            if t % esteps == 0:
                ss.append(helper(t))
        else:
            if t %(20*esteps) == 0 or (t == T):
                ss.append(helper(t))
    return ss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-args', '-d',  type=str, help='dataset name, augmentation')
    parser.add_argument('--model-args', '-m', type=str, help='model name, model aruguments, batch normalization')
    parser.add_argument('--optim-args', '-o', type=str, help='batch size, batch seed, learning rate, optimizer, weight decay')
    parser.add_argument('--init-args', '-i', type=str, help='start from corner')

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
    ds = get_data(data_args, dev=dev)
    T = args.epochs * ds['x'].shape[0] // args.bs
    optim_args['T'] = T

    if args.batch_seed < 0:
        fix_batch = np.zeros(2)
    else:
        setup(args.batch_seed)
        fix_batch = np.random.randint(ds['x'].shape[0], size=(T, args.bs))

    setup(args.seed)

    m = get_model(model_args, dev=dev)
    optimizer, scheduler = get_opt(optim_args, m)

    m = get_init(init_args, m)
    ss = fit(m, ds, T=T, bs=optim_args['bs'], autocast=optim_args['autocast'], opt=optimizer, sched=scheduler, fix_batch=fix_batch)

    th.save({"data": ss, "configs": args}, os.path.join(root, fn+'.p'))

if __name__ == '__main__':
    # python runner.py -d ./configs/data/cifar10-full.yaml -m ./configs/model/convmixer-256-8-5-2.yaml -o ./configs/optim/adamw-500.yaml -i ./configs/init/normal.yaml 
    # gives 88% acc
    main()
