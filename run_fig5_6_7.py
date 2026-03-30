"""
Appendix Figures 5, 6, 7: Hyperparameter sweeps for behavioral plasticity.
Figure 5: L1 regularization sweep (0.001, 0.005, 0.01, 0.05)
Figure 6: Moving average window sweep (5, 10, 20) for Taylor and MLP
Figure 7: Input firing mean sweep (0.25, 0.5, 0.75, 1.0, 1.25)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time
from src.plasticity_rules import TaylorRule4Var, MLPRule
from src.network import simulate_behavior_model, behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global progress counter
TOTAL_RUNS = 60  # 12 + 18 + 30
COMPLETED = 0
T_START = None

def progress():
    global COMPLETED
    COMPLETED += 1
    elapsed = time.time() - T_START
    per_run = elapsed / COMPLETED
    remaining = (TOTAL_RUNS - COMPLETED) * per_run
    pct = 100.0 * COMPLETED / TOTAL_RUNS
    print(f"  [{COMPLETED}/{TOTAL_RUNS}] {pct:.0f}% done | "
          f"Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min remaining")

def gen_gt(seed, n_input=2, n_hidden=10, traj_len=240,
           firing_mean=0.75, noise_var=0.015, ma_window=10):
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
        if c==1: R=int(np.random.binomial(1,rs[t,oi]))
        rews.append(float(R)); r=R-ravg
        a=1.0/ma_window; ravg=(1-a)*ravg+a*R
        dW=x.unsqueeze(0).expand(n_hidden,-1)*r; W=W+lr*dW
    return {'inputs':torch.stack(inps),'choices':torch.tensor(chs,dtype=torch.float32),
            'rewards':torch.tensor(rews,dtype=torch.float32),'weights':torch.stack(wts),
            'hidden':torch.stack(hids),'W0':W0}

def r2(yt,yp):
    sr=np.sum((yt-yp)**2); st=np.sum((yt-yt.mean())**2); return 1.0-sr/(st+1e-10)

def train_eval(trd, evd, mtype='taylor', ne=400, lr=1e-3, l1=1e-2, seed=0, maw=10):
    torch.manual_seed(seed)
    rule = (TaylorRule4Var(1e-5) if mtype=='taylor' else MLPRule(4,10)).to(DEVICE)
    opt = torch.optim.Adam(rule.parameters(), lr=lr)
    nt = len(trd)
    
    for ep in range(ne):
        d = trd[ep % nt]
        i = d['inputs'].to(DEVICE); rw = d['rewards'].to(DEVICE)
        c = d['choices'].to(DEVICE); w0 = d['W0'].to(DEVICE)
        opt.zero_grad()
        p, wl, h = simulate_behavior_model(w0, i, rw, rule, moving_avg_window=maw)
        e = 1e-7; pc = p.clamp(e, 1-e)
        bce = -torch.mean(c * torch.log(pc) + (1-c) * torch.log(1-pc))
        loss = bce + (l1 * rule.coeffs.abs().sum() if mtype=='taylor' and l1 > 0 else 0)
        
        # NaN / divergence protection
        if torch.isnan(loss) or loss.item() > 100:
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(rule.parameters(), 1.0)
        opt.step()
    
    rule.eval()
    r2s = []
    with torch.no_grad():
        for d in evd:
            i = d['inputs'].to(DEVICE); rw = d['rewards'].to(DEVICE)
            w0 = d['W0'].to(DEVICE)
            p, wl, h = simulate_behavior_model(w0, i, rw, rule, moving_avg_window=maw)
            r2s.append(r2(d['weights'].numpy().flatten(),
                         torch.stack(wl).cpu().numpy().flatten()))
    return np.median(r2s)


# ============ Figure 5: L1 Regularization ============

def run_fig5(sp='figures/figure5.png'):
    print("\n=== Figure 5: L1 Regularization ===")
    l1s = [0.001, 0.005, 0.01, 0.05]; ns = 3
    res = {l: [] for l in l1s}
    
    for l in l1s:
        for s in range(ns):
            d = [gen_gt(s*100+i) for i in range(25)]
            res[l].append(train_eval(d[:18], d[18:], 'taylor', 400, l1=l, seed=s*1000))
            progress()
        print(f"    L1={l}: R2_w = {np.mean(res[l]):.3f} +/- {np.std(res[l]):.3f}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([res[l] for l in l1s], labels=[str(l) for l in l1s],
                     patch_artist=True, widths=0.6)
    for p in bp['boxes']:
        p.set_facecolor('#2E7D32'); p.set_alpha(0.6)
    ax.set_xlabel('L1 Regularization'); ax.set_ylabel(r'$R^2$ Weights')
    ax.set_title('L1 Regularization', fontweight='bold'); ax.set_ylim(0, 1.05)
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    plt.tight_layout(); plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved {sp}")


# ============ Figure 6: Moving Average Window ============

def run_fig6(sp='figures/figure6.png'):
    print("\n=== Figure 6: Moving Average Window ===")
    wins = [5, 10, 20]; ns = 3
    res = {m: {w: [] for w in wins} for m in ['taylor', 'mlp']}
    
    for mt in ['taylor', 'mlp']:
        for w in wins:
            for s in range(ns):
                d = [gen_gt(s*100+i, ma_window=w) for i in range(25)]
                res[mt][w].append(train_eval(d[:18], d[18:], mt, 400, seed=s*1000, maw=w))
                progress()
            print(f"    {mt} win={w}: R2_w = {np.mean(res[mt][w]):.3f}")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ['#43A047', '#2E7D32', '#1B5E20']; wd = 0.25
    for i, w in enumerate(wins):
        off = (i - 1) * wd
        for j, mt in enumerate(['taylor', 'mlp']):
            bp = ax.boxplot([res[mt][w]], positions=[j+off], widths=wd*0.8, patch_artist=True)
            for p in bp['boxes']:
                p.set_facecolor(cols[i]); p.set_alpha(0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Taylor', 'MLP'])
    ax.set_ylabel(r'$R^2$ Weights')
    ax.set_title('Moving Average Window', fontweight='bold'); ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(facecolor=cols[i], alpha=0.7, label=str(w))
                       for i, w in enumerate(wins)], title='Window')
    plt.tight_layout(); plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved {sp}")


# ============ Figure 7: Input Firing Mean ============

def run_fig7(sp='figures/figure7.png'):
    print("\n=== Figure 7: Input Firing Mean ===")
    fms = [0.25, 0.5, 0.75, 1.0, 1.25]; ns = 3
    res = {m: {f: [] for f in fms} for m in ['taylor', 'mlp']}
    
    for mt in ['taylor', 'mlp']:
        for fm in fms:
            for s in range(ns):
                d = [gen_gt(s*100+i, firing_mean=fm) for i in range(25)]
                res[mt][fm].append(train_eval(d[:18], d[18:], mt, 400, seed=s*1000))
                progress()
            print(f"    {mt} fm={fm}: R2_w = {np.mean(res[mt][fm]):.3f}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ['#A5D6A7', '#66BB6A', '#43A047', '#2E7D32', '#1B5E20']; wd = 0.15
    for i, fm in enumerate(fms):
        off = (i - 2) * wd
        for j, mt in enumerate(['taylor', 'mlp']):
            bp = ax.boxplot([res[mt][fm]], positions=[j+off], widths=wd*0.8, patch_artist=True)
            for p in bp['boxes']:
                p.set_facecolor(cols[i]); p.set_alpha(0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Taylor', 'MLP'])
    ax.set_ylabel(r'$R^2$ Weights')
    ax.set_title('Input Firing Mean', fontweight='bold'); ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(facecolor=cols[i], alpha=0.8, label=str(fm))
                       for i, fm in enumerate(fms)], title='Firing Mean')
    plt.tight_layout(); plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved {sp}")


if __name__ == '__main__':
    print("=" * 60)
    print("Appendix Figures 5, 6, 7: Hyperparameter Sweeps")
    print(f"Device: {DEVICE}")
    print(f"Total training runs: {TOTAL_RUNS}")
    print("=" * 60)
    
    T_START = time.time()
    run_fig5()
    run_fig6()
    run_fig7()
    
    total = time.time() - T_START
    print(f"\n{'='*60}")
    print(f"All done! Total time: {total/60:.1f} minutes ({total/3600:.1f} hours)")
