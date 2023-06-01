from pcutils import *

def data(κ=0,N=100,d=16,c=1,
         h=16,seed=42):
    setup(seed)

    x=th.randn(N,d)
    u,σ,v=th.svd(x);σ=th.logspace(0,-κ,d);
    x=u@th.diag(σ)@v
    t=mlp_t(hs=[d,h,h,c])
    if c>1:
        y=t(x).argmax(1)
        #y=x.argmax(1)
    else:
        y=t(x)
    return {'x':x,'y':y.detach()}

def fit(x,y,nn,T=100,bsz=64,
        loss=F.cross_entropy,
        sgd=partial(th.optim.SGD,lr=0.1,
            nesterov=True,momentum=0.9),
        seed=42,
        verbose=True):
    setup(seed)
    m=nn()
    opt=sgd(m.parameters())
    if T > 0:
        sched=th.optim.lr_scheduler.OneCycleLR(opt,
                max_lr=opt.param_groups[0]['lr'],total_steps=T)
    tmp=th.arange(len(y))

    ss=[]
    Ts=list(np.arange(0,T//10,int(max(1,T//1000))))+ \
            list(np.arange(T//10,T,int(max(1,T//100))))
    for k in range(T):
        i = np.random.choice(tmp,bsz)
        opt.zero_grad()
        loss(m(x[i]),y[i]).backward()
        opt.step()
        sched.step()
        if (k in Ts) and verbose:
            with th.no_grad():
                yh=m(x);f=loss(yh,y,reduction='none');
                h,_=ddf(x,m,method='ntk')
                ss.append({'t':k, 'yh':yh.numpy(), 'f':f.numpy(),
                           'k':h.numpy(),
                           'seed':seed})
            print(f'[{seed:02d}][{k:05d}] f: {f.numpy().mean():.3f}')
    return ss

def test():
    N=1000;d=16;c=1;h=32;
    ds=data(κ=0,N=N,d=d,c=c)
    x=ds['x'];y=ds['y'];
    d=x.shape[-1];
    m=mlp_t(hs=[d,h,c],bn=None)

    h0,_=ddf(x,m,method='ntk')
    h1,_=ddf(x,m,method='fisher-block-diagonal')
    h2,_=ddf(x,m,method='fisher-full')
    h3,_=ddf(x,m,y,method='hessian-full',c=F.cross_entropy)
    h4,_=ddf(x,m,y,method='hessian-block-diagonal',c=F.cross_entropy)


N=100;d=16;c=1;h=8;
ds=data(κ=5,N=N,d=d,c=c)
x,y=ds['x'],ds['y']
r=fit(x,y,partial(mlp_t,hs=[d,h,c],bn=None),
      T=5000,loss=F.mse_loss)

def P(v,M):
    return M @ np.linalg.pinv(M.T @ M) @ M.T @ v

v=[];vr=[];vdet=[];dk=[];k=[];
for t in range(len(r)):
    kt=r[t]['k'];ktm=r[t-1]['k'];k0=r[0]['k'];kT=r[-1]['k']

    dk.append(np.linalg.eigh(kt-ktm))
    v.append(np.linalg.eigh(kt-k0))
    vr.append(np.linalg.eigh(kt-kT))
    vdet.append(np.linalg.slogdet(P(kt,k0))[1])
    k.append(np.linalg.eigh(kt))


plt.figure(1);plt.clf();
plt.plot([np.sum(t[0]) for t in k], label='<λ(K(t))>')
plt.plot([np.sum(np.abs(t[0])) for t in dk], label='<λ(K(t)-K(t-1))>')
plt.plot([np.sum(np.abs(t[0])) for t in v],label='<λ(K(t)-K(0))>')
plt.plot([np.sum(np.abs(t[0])) for t in vr],label='<λ(K(t)-K(T))>')
plt.plot([np.mean(t['f']) for t in r], label='<mse>')
plt.legend()
plt.yscale('log')
