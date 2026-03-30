"""
Table 2: Scalability analysis.
- Vary trajectory length (30, 60, 120, 240, 480, 960, 1920) with hidden=10
- Vary hidden layer size (10, 50, 100, 500, 1000) with traj_len=240
Ground truth: dw = x_j * r, Taylor series, 3 seeds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import time
from src.plasticity_rules import TaylorRule4Var
from src.network import simulate_behavior_model, behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_gt(seed, n_input=2, n_hidden=10, traj_len=240, firing_mean=0.75, noise_var=0.015, ma_win=10):
    torch.manual_seed(seed); np.random.seed(seed)
    rp = np.array([[0.2,0.8],[0.9,0.1],[0.2,0.8]])
    bs = traj_len // 3
    rs = np.concatenate([np.tile(rp[b], (bs,1)) for b in range(3)])
    W = torch.randn(n_hidden, n_input)*0.01; W0=W.clone(); lr=1.0/n_input
    inps,chs,rews,wts,hids=[],[],[],[],[]
    ravg=0.0
    for t in range(traj_len):
        oi=int(torch.randint(0,2,(1,)).item())
        x=torch.zeros(n_input); x[oi]=firing_mean
        x=x+torch.randn(n_input)*(noise_var**0.5); inps.append(x)
        h,_,p=behavior_forward(W,x,n_hidden); hids.append(h.clone()); wts.append(W.clone())
        c=int(torch.bernoulli(torch.tensor(p.item())).item()); chs.append(c)
        R=0
        if c==1: R=int(np.random.binomial(1,rs[t%len(rs),oi]))
        rews.append(float(R)); r=R-ravg
        a=1.0/ma_win; ravg=(1-a)*ravg+a*R
        dW=x.unsqueeze(0).expand(n_hidden,-1)*r; W=W+lr*dW
    return {'inputs':torch.stack(inps),'choices':torch.tensor(chs,dtype=torch.float32),
            'rewards':torch.tensor(rews,dtype=torch.float32),'weights':torch.stack(wts),
            'hidden':torch.stack(hids),'W0':W0}

def r2(yt,yp):
    sr=np.sum((yt-yp)**2); st=np.sum((yt-yt.mean())**2); return 1.0-sr/(st+1e-10)

def pde(probs, choices):
    e=1e-7; p=np.clip(probs,e,1-e); c=np.array(choices)
    llm=np.sum(c*np.log(p)+(1-c)*np.log(1-p))
    pn=np.clip(c.mean(),e,1-e); lln=np.sum(c*np.log(pn)+(1-c)*np.log(1-pn))
    return 100.0*(1.0-(-2*llm)/(-2*lln+1e-10))

def train_eval_full(trd, evd, n_epochs=400, lr=1e-3, l1=1e-2, seed=0):
    torch.manual_seed(seed)
    rule=TaylorRule4Var(1e-5).to(DEVICE)
    opt=torch.optim.Adam(rule.parameters(),lr=lr); nt=len(trd)
    for ep in range(n_epochs):
        d=trd[ep%nt]; i=d['inputs'].to(DEVICE); rw=d['rewards'].to(DEVICE)
        c=d['choices'].to(DEVICE); w0=d['W0'].to(DEVICE)
        opt.zero_grad()
        p,wl,h=simulate_behavior_model(w0,i,rw,rule)
        e=1e-7; pc=p.clamp(e,1-e)
        bce=-torch.mean(c*torch.log(pc)+(1-c)*torch.log(1-pc))
        loss=bce+l1*rule.coeffs.abs().sum()
        loss.backward(); torch.nn.utils.clip_grad_value_(rule.parameters(),1.0); opt.step()
    rule.eval(); r2w,r2a,pdes=[],[],[]
    with torch.no_grad():
        for d in evd:
            i=d['inputs'].to(DEVICE); rw=d['rewards'].to(DEVICE); w0=d['W0'].to(DEVICE)
            p,wl,h=simulate_behavior_model(w0,i,rw,rule)
            mw=torch.stack(wl).cpu().numpy(); gw=d['weights'].numpy()
            mh=h.cpu().numpy(); gh=d['hidden'].numpy()
            r2w.append(r2(gw.flatten(),mw.flatten()))
            r2a.append(r2(gh.flatten(),mh.flatten()))
            pdes.append(pde(p.cpu().numpy(),d['choices'].numpy()))
    return np.median(r2w), np.median(r2a), np.median(pdes)

if __name__=='__main__':
    print("="*60); print("Table 2: Scalability Analysis"); print(f"Device: {DEVICE}"); print("="*60)
    t0=time.time(); ns=3
    
    # Part 1: Vary trajectory length (hidden=10)
    traj_lens = [30, 60, 120, 240, 480, 960, 1920]
    print("\n--- Trajectory Length (hidden=10) ---")
    print(f"{'TrajLen':>8} {'R2_W':>8} {'R2_A':>8} {'%Dev':>8}")
    tl_results = {}
    for tl in traj_lens:
        r2ws,r2as,pds=[],[],[]
        for s in range(ns):
            print(f"  tl={tl} seed={s}",end='\r')
            d=[gen_gt(s*100+i,traj_len=tl) for i in range(25)]
            rw,ra,pd=train_eval_full(d[:18],d[18:],n_epochs=400,seed=s*1000)
            r2ws.append(rw); r2as.append(ra); pds.append(pd)
        tl_results[tl] = (np.mean(r2ws),np.mean(r2as),np.mean(pds))
        print(f"  {tl:>6}: {np.mean(r2ws):>8.2f} {np.mean(r2as):>8.2f} {np.mean(pds):>8.2f}")
    
    # Part 2: Vary hidden size (traj_len=240)
    hidden_sizes = [10, 50, 100, 500, 1000]
    print("\n--- Hidden Layer Size (traj_len=240) ---")
    print(f"{'Hidden':>8} {'R2_W':>8} {'R2_A':>8} {'%Dev':>8}")
    hs_results = {}
    for nh in hidden_sizes:
        r2ws,r2as,pds=[],[],[]
        for s in range(ns):
            print(f"  hidden={nh} seed={s}",end='\r')
            d=[gen_gt(s*100+i,n_hidden=nh) for i in range(25)]
            rw,ra,pd=train_eval_full(d[:18],d[18:],n_epochs=400,seed=s*1000)
            r2ws.append(rw); r2as.append(ra); pds.append(pd)
        hs_results[nh] = (np.mean(r2ws),np.mean(r2as),np.mean(pds))
        print(f"  {nh:>6}: {np.mean(r2ws):>8.2f} {np.mean(r2as):>8.2f} {np.mean(pds):>8.2f}")
    
    # Print final table
    print("\n" + "="*60)
    print("TABLE 2 RESULTS")
    print("="*60)
    print("\nTrajectory Length (hidden=10):")
    print(f"{'':>12}", end='')
    for tl in traj_lens: print(f"{tl:>8}", end='')
    print()
    for metric, idx, label in [('R2_W',0,'R² Weights'),('R2_A',1,'R² Activity'),('%Dev',2,'% Deviance')]:
        print(f"{label:>12}", end='')
        for tl in traj_lens: print(f"{tl_results[tl][idx]:>8.2f}", end='')
        print()
    
    print(f"\nHidden Layer Size (traj_len=240):")
    print(f"{'':>12}", end='')
    for nh in hidden_sizes: print(f"{nh:>8}", end='')
    print()
    for metric, idx, label in [('R2_W',0,'R² Weights'),('R2_A',1,'R² Activity'),('%Dev',2,'% Deviance')]:
        print(f"{label:>12}", end='')
        for nh in hidden_sizes: print(f"{hs_results[nh][idx]:>8.2f}", end='')
        print()
    
    print(f"\nTotal: {time.time()-t0:.1f}s")
