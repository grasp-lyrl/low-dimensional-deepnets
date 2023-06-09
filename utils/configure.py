import random
from copy import deepcopy
from functools import partial
import yaml
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch as th
import torchvision as thv
import torchvision.transforms as transforms
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
import os
from types import SimpleNamespace

from runner_corner import fit
import networks, init


def setup(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True


def get_configs(fname):
    with open(fname, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def save_configs(c, fname):
    with open(fname, 'w') as f:
        yaml.safe_dump(c, f)


def get_model(model_args, dev='cpu'):
    m, model_args = model_args['m'], model_args['model_args']
    return getattr(networks, m)(**model_args).to(dev)


def get_opt(optim_args, model):
    opt, opt_args = optim_args['opt'], optim_args['opt_args']
    optimizer = getattr(th.optim, opt)(model.parameters(), **opt_args)

    sched, sched_args = optim_args['scheduler'], optim_args['sched_args']
    if sched == 'cosine_with_warmup':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=optim_args['T']//2,
                                                  max_lr=opt_args['lr'], **sched_args)
    elif sched == 'cosine':
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optim_args['T'])
    else:
        scheduler = getattr(th.optim.lr_scheduler, sched)(**sched_args)

    return optimizer, scheduler


def get_init(init_args, model, dev='cuda', data=None, save_fn='', init_seed=42):
    init_fn = init_args['init_fn']
    if init_fn == 'corner':
        corner = init_args['corner']
        if corner == "normal":
            return model
        else:
            opt_args = init_args['init_opts']
            opt, sched = get_opt(opt_args, model)
            if corner == "uniform":
                fn = os.path.join('/home/ubuntu/ext_vol/data', 
                                  f"{init_args['corner_data']}_{init_args['corner']}_{init_seed}.p")
                if os.path.exists(fn):
                    print('loading data from ', fn)
                    ds_init = th.load(fn)
                else:
                    ds_init = relabel_data(data, frac=1, seed=init_seed)
                    th.save(ds_init, fn)
            elif corner == "subsample":
                ds_init = get_data(init_args['init_data'], dev=dev)
            init_ss = fit(model, ds_init, epochs=opt_args['epochs'], bs=opt_args['bs'], save_init=0,
                        autocast=opt_args['autocast'], opt=opt, sched=sched)
            if save_fn:
                th.save({'data': init_ss, 'configs': SimpleNamespace(**{**init_args})}, save_fn)
            return model
    elif init_fn == 'perturbed_normal':
        weights = deepcopy(model.state_dict())
        fn = os.path.join('/home/ubuntu/ext_vol/init', f"{save_fn}{init_seed}_normal_{init_args['mean']}_{init_args['std']}.p")
        if os.path.exists(fn):
            print('loading initial weights from ', fn)
            model.load_state_dict(th.load(fn))
        else:
            setup(init_seed)
            model.apply(partial(init.fixed_var_normal.init_weights, mean=init_args['mean'], std=init_args['std']))
            th.save(model.state_dict(), fn)
        for (n, p) in model.named_parameters():
            p.data += weights[n]
        return model
    else:
        init_fn = getattr(init, init_fn)
        model.apply(partial(init_fn.init_weights, **init_args['init_fn_args']))
        return model


def get_data(data_args={'data': 'CIFAR10', 'aug': 'none', 'sub_sample': 0}, resize=1):
    name = data_args['data']
    assert name in ['CIFAR10', 'CIFAR100', 'MNIST', 'synthetic', 'load_existing']

    if name == 'synthetic':
        c = data_args['c']
        label_model = data_args['label_model']
        fn = os.path.join('/home/ubuntu/ext_vol/data/synthetic/', f'{label_model}-{c}.pt')
        if os.path.exists(fn):
            print("loading from ", fn)
            ds = th.load(fn)
        else:
            seed = 5
            setup(seed)
            num_train = data_args['num_train']
            num_val = data_args['num_val']
            d = np.prod(data_args['dshape'])
            b = 50*c if c > 0 else 1
            evals = b*th.exp(-c*th.arange(d))
            data_train = (th.randn(num_train, d) @ th.diag(th.sqrt(evals / d))).reshape(-1, *data_args['dshape'])
            data_val= (th.randn(num_val, d) @ th.diag(th.sqrt(evals / d))).reshape(-1, *data_args['dshape'])
            
            model_args = get_configs(
                f"/home/ubuntu/ext_vol/inpca/configs/model/{label_model}.yaml")
            m = get_model(model_args, dev='cuda')
            m_init = data_args.get('label_model_init', '')
            if m_init:
                label_model_init = get_configs(
                    f"/home/ubuntu/ext_vol/inpca/configs/init/{m_init}.yaml"
                )
                m = get_init(label_model_init, m)
            targets_train = th.zeros(len(data_train))
            targets_val = th.zeros(len(data_val))
            for i in range(0, max(num_train, num_val), 500):
                if i < num_train:
                    targets_train[i:i +
                            500] = th.argmax(m(data_train[i:i+500, :].cuda()), axis=1).cpu()
                if i < num_val:
                    targets_val[i:i+500] = th.argmax(m(data_val[i:i+500, :].cuda()), axis=1).cpu()
            ds = {'train': SyntheticData(x=data_train, y=targets_train.long()),
                'val': SyntheticData(x=data_val, y=targets_val.long())}
            th.save(ds, fn)
    elif name == 'load_existing':
        ds = th.load(data_args['fn']) 
    else:
        aug, sub_sample = data_args['aug'], data_args['sub_sample']
        assert aug in ['none', 'simple', 'full']
        if aug == 'full':
            aug_args = data_args['aug_args']

        f = getattr(thv.datasets, name)

        if name in ['CIFAR10', 'CIFAR100']:
            sz = 32//resize
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)
            if aug == 'simple':
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)
                ])
            elif aug == 'full':
                transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(
                        aug_args['scale'], 1.0), ratio=(1.0, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandAugment(num_ops=aug_args['ra_n'], magnitude=aug_args['ra_m']),
                    transforms.ColorJitter(
                        aug_args['jitter'], aug_args['jitter'], aug_args['jitter']),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                    transforms.RandomErasing(p=aug_args['reprob'])
                ])
            elif aug == 'none':
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std)
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std)
            ])
        elif name == 'STL10':
            sz = 96//resize
        elif name == 'MNIST':
            sz = 28//resize
        else:
            assert False

        ds = {'train': f('/home/ubuntu/ext_vol/data', train=True, download=False, transform=transform_train),
            'val': f('/home/ubuntu/ext_vol/data', train=False, download=False, transform=transform_test)}

        # subsample the dataset, make sure it is balanced
        if sub_sample > 0:
            y = th.tensor(ds['train'].targets).long()
            l = th.max(y)+1  # number of classes
            # number of samples each class
            ss = th.div(sub_sample, l, rounding_mode='trunc')
            idxs = th.cat([th.where(y == i)[0][:ss] for i in range(l)])
            ds['train'] = th.utils.data.Subset(ds['train'], idxs)
            if name == "MNIST":
                yv = th.tensor(ds['test'].targets).long()
                idxs = th.cat([th.where(yv == i)[0][:850] for i in range(l)])
                ds['test'] = th.utils.data.Subset(ds['test'], idxs)
        
        noise_rate = data_args.get('noise_rate', 0)
        if noise_rate > 0:
            y = th.tensor(ds['train'].targets).long()
            l = th.max(y)+1  # number of classes
            ss = [math.floor(noise_rate*sum(y==i)) for i in range(l)]
            idxs = th.cat([th.where(y == i)[0][:ss[i]] for i in range(l)])
            y[idxs] = th.randint(0, l, [len(idxs)])
            ds['train'].targets = y.tolist()

    return ds


def relabel_data(ds, frac=1, seed=0):
    ds_new = deepcopy(ds)
    targets = np.array(ds['train'].targets)
    n = len(targets)
    n_rand = int(n*frac)
    idx = np.random.RandomState(seed=seed).permutation(n)[:n_rand]
    y_rand = np.random.RandomState(seed=seed).randint(0, 10, (n_rand, ))
    targets[idx] = y_rand
    ds_new['train'].targets = list(targets)
    return ds_new


class CosineAnnealingWarmupRestarts(_LRScheduler):
    # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: th.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps *
                            (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - \
                        int(self.first_cycle_steps *
                            (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * \
                        self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



class SyntheticData(Dataset):
    def __init__(self, x, y): 
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.y)
