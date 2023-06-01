import numpy as np,\
        scipy as sp,\
        pandas as pd,\
        torch as th,\
        torchvision as thv,\
        torch.nn as nn,\
        torch.nn.functional as F,\
        os,random,inspect
from functools import partial
import torch.func as FF

#from backpack import extend, backpack
#from backpack.extensions import KFAC,KFLR,KFRA

#from torch_utils.ops import conv2d_gradfix
#from torch_utils.ops import grid_sample_gradfix
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.benchmark=True
th.backends.cudnn.allow_tf32 = True
#conv2d_gradfix.enabled = True
#grid_sample_gradfix.enabled = True

import matplotlib.pyplot as plt
plt.ion()

import seaborn as sns
sns.set(context='notebook',
        style='ticks',
        font_scale=1,
        rc={'axes.grid':True,
            'grid.color':'.9',
            'grid.linewidth':0.75})

num_params=lambda m: sum([p.numel() for p in m.parameters()])

def setup(s):
    th.manual_seed(s)
    random.seed(s)
    np.random.seed(s)

class mlp_t(nn.Module):
    def __init__(s,hs=[64],
                 act=nn.ReLU,
                 bn=partial(nn.BatchNorm1d,affine=True)):
        super().__init__()
        ll=[]
        for h1,h2 in zip(hs[:-2],hs[1:-1]):
            ll += [
                nn.Linear(h1,h2),
                act(),
                bn(h2) if bn is not None else None]
        ll.append(nn.Linear(hs[-2],hs[-1]))
        ll=[_ for _ in ll if _ is not None]
        s.m = nn.Sequential(*ll)
    def forward(s,x):
        return s.m(x).squeeze()

class lenet_t(nn.Module):
    def __init__(s,ydim=2):
        c1=10;c2=20;c3=100;
        super().__init__()

        bn1, bn2 = nn.BatchNorm1d, nn.BatchNorm2d
        def convbn(ci,co,ksz,psz):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                bn2(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz))

        class View(nn.Module):
            def __init__(s,o):
                super().__init__()
                s.o = o
            def forward(s,x):
                return x.view(-1, s.o)

        s.m = nn.Sequential(
            convbn(1,c1,5,3),
            convbn(c1,c2,5,2),
            View(c2*2*2),
            nn.Linear(c2*2*2, c3),
            bn1(c3),
            nn.ReLU(True))
        s.fc=nn.Linear(c3,ydim)

    def forward(s,x):
        if len(x.shape)==3: x=x.unsqueeze(1)
        f=s.m(x)
        return s.fc(f),f

def fit(x,y,nn,T=1000,bsz=64,
        loss=F.mse_loss,
        sgd=partial(th.optim.SGD,lr=0.1,
            nesterov=True,momentum=0.9),
        use_sched=True,
        verbose=True):
    if inspect.isclass(nn):
        m=nn()
    else:
        m=nn
    opt=sgd(m.parameters())
    if T > 0 and use_sched:
        sched=th.optim.lr_scheduler.OneCycleLR(opt,
                max_lr=opt.param_groups[0]['lr'],total_steps=T)
    tmp=th.arange(len(y))
    fs=[]
    for k in range(T):
        i = np.random.choice(tmp,bsz)
        def helper():
            opt.zero_grad()
            yh=m(x[i])
            f=loss(yh,y[i])
            f.backward()
            return f

        opt.step(helper)
        with th.no_grad():
            yh=m(x[i])
            f=loss(yh,y[i])
            fs.append(f.item())
        if use_sched:
            sched.step()
        if (k%(max(T//10,1))==0) and verbose:
            print(f'[{k:05d}] f: {np.mean(fs):.3f}')
            fs=[]
    return m

def pred(x,m):
    with th.no_grad():
        yh=m(x)
    return yh

def test_power(a,n=2):
    es=[];evs=[];
    ϵ=1e-3;T=10;
    for i in range(n):
        e=None
        v=th.randn(a.shape[0]);v=v/th.norm(v);
        for t in range(T):
            for vv in evs:
                v=v-th.dot(v,vv)*vv
            v=v/th.norm(v)
            ee = th.dot(v,a@v).item()
            v=a@v;v=v/th.norm(v);
            if e is None or abs(e-ee)/(abs(e)+1e-6)>ϵ:
                e=ee
            else:
                print(f'{i:02d}: [{t:03d}] e:{e:.3f}')
                break
        es.append(e)
        evs.append(v)
    return es,evs

def ggn(x,y,m,loss=th.nn.MSELoss()):
    mm=extend(m.m);mse=extend(loss);
    f=mse(mm(x),y)

    with backpack(KFAC(mc_samples=100)):
        f.backward()

    es=[]
    for name,p in mm.named_parameters():
        e=[]
        for kk in p.kfac:
            e.append(th.real(th.linalg.eig(kk)[0]))
        if len(e)==2:
            es.append((e[0].view(-1,1) @ e[1].view(1,-1)).view(-1))
        else:
            es.append(e[0])
    e=th.cat(es)
    idx=th.argsort(th.abs(e),descending=True);e=e[idx];
    return e

def ddf(x,m,y=None,c=F.mse_loss,method='ntk',
        σ=1):
    N=x.shape[0]
    if method=='ntk':
        assert 'mse_loss' in str(c),f'c is {c}'
        def f(w,x):
            return FF.functional_call(m,w,x)
        j=FF.jacrev(f)(dict(m.named_parameters()),x)
        t=th.concatenate([j[k].flatten(1) for k in j.keys()],-1).detach()
        return th.einsum('ia,ja->ij',t,t),j.keys()

    if 'fisher' in method:
        def f(w,x):
            yh=FF.functional_call(m,w,x)
            if yh.ndim==1:
                # monte-carlo sampling for real-valued outputs
                C=16
                yh=yh.view(N,-1).expand(N,C)
                y=th.normal(yh,σ)
                p=F.gaussian_nll_loss(y,yh,σ**2*th.ones_like(yh),reduction='none')
                p=p/p.sum(-1,keepdim=True)
            else:
                C=p.shape[-1]
                p=F.softmax(p,dim=-1)
            return p

        p=f(dict(m.named_parameters()),x).detach()
        C=p.shape[1]
        p=th.sqrt(p).view(N,C,-1)
        j=FF.jacrev(f)(dict(m.named_parameters()),x)
        if method=='fisher-full':
            t=th.concatenate([j[k].flatten().view(N,C,-1) for k in j.keys()],-1).detach()
            return th.einsum('Nya,Nyb->ab',p*t,p*t)/N, j.keys()
        if method=='fisher-block-diagonal':
            j={k:v.view(N,C,-1) for k,v in j.items()}
            t={k:p.expand_as(v)*v.detach() for k,v in j.items()}
            return {k:th.einsum('Nya,Nyb->ab',v,v)/N for k,v in t.items()},j.keys()

    if 'hessian' in method:
        assert y is not None,'y is None'
        C=y.shape[-1]
        def f(w,x):
            return c(FF.functional_call(m,w,x).squeeze(),y)
        h=FF.hessian(f)(dict(m.named_parameters()),x)
        if method=='hessian-full':
            n=num_params(m)
            h1=th.cat([th.cat([e.flatten() for _,e in hh.items()]) for _,hh in h.items()])
            return h1.reshape(n,n).detach(),h.keys()
        if method=='hessian-block-diagonal':
            return {k:h[k][k].squeeze().detach() for k in h},h.keys()

