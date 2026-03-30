"""
Appendix Figures 8 & 9: Held-out validation on real fly data.

Figure 8: Train on first x% of trajectory, test on remaining (100-x)%.
          Sweep x from 0.05 to 1.0.
Figure 9: Weight decay coefficient (θ_001) for flies with good vs poor test performance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import scipy.io as sio
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.plasticity_rules import FlyPlasticityWithW
from src.network import behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_fly_data(data_dir='./data/', n_flies=18):
    fly_data = []
    for i in range(1, n_flies + 1):
        path = os.path.join(data_dir, f'Fly{i}.mat')
        if not os.path.exists(path): continue
        d = sio.loadmat(path)
        X = d['X']; Y = d['Y'].flatten(); R = d['R'].flatten()
        n_rej = len(Y) - int(Y.sum())
        if n_rej < 5: continue  # Skip accept-only flies
        fly_data.append({
            'odors': torch.tensor(X, dtype=torch.float32),
            'decisions': torch.tensor(Y, dtype=torch.float32),
            'rewards': torch.tensor(R, dtype=torch.float32),
            'n_timesteps': len(Y), 'n_trials': int(Y.sum()), 'fly_id': i,
        })
    return fly_data


def gen_inputs(fly_datum, seed=0, firing_mean=0.75, noise_var=0.015):
    torch.manual_seed(seed)
    odors = fly_datum['odors']
    noise = torch.randn_like(odors) * (noise_var ** 0.5)
    return odors * firing_mean + noise


def simulate_fly(W0, inputs, decisions, rewards, rule, n_hidden=10, use_er=True):
    """Simulate with variable-length trials."""
    n_t = inputs.shape[0]; n_in = inputs.shape[1]
    W = W0.clone(); lr = 1.0 / n_in
    ravg = torch.tensor(0.0, device=W0.device)
    probs = []; tidx = 0
    for t in range(n_t):
        x = inputs[t]
        h, _, p = behavior_forward(W, x, n_hidden)
        probs.append(p)
        if decisions[t] > 0.5:
            R = rewards[tidx] if tidx < len(rewards) else torch.tensor(0.0, device=W0.device)
            tidx += 1
            r = (R - ravg) if use_er else R
            ravg = ((1-0.1)*ravg + 0.1*R).detach()
            dW = rule(x, h, W, r)
            dW = dW.clamp(-1.0, 1.0)
            W = W + lr * dW
            W = W.clamp(-5.0, 5.0)
    return torch.stack(probs)


def pde(probs, decisions):
    e = 1e-7; p = np.clip(probs, e, 1-e); d = np.array(decisions)
    llm = np.sum(d*np.log(p) + (1-d)*np.log(1-p))
    pn = np.clip(d.mean(), e, 1-e)
    lln = np.sum(d*np.log(pn) + (1-d)*np.log(1-pn))
    return 100.0 * (1.0 - (-2*llm)/(-2*lln+1e-10))


def train_fly_split(fly_datum, train_frac, n_epochs=250, n_samp=10, lr=1e-3, seed=0):
    """Train on first train_frac of data, evaluate on rest."""
    n_t = fly_datum['n_timesteps']
    split = int(n_t * train_frac)
    if split < 10 or split >= n_t - 5:
        return None, None, None
    
    # Split data
    train_dec = fly_datum['decisions'][:split].to(DEVICE)
    test_dec = fly_datum['decisions'][split:].to(DEVICE)
    
    # Split rewards: count accepts in train portion to split R
    n_train_accepts = int(train_dec.sum().item())
    train_rew = fly_datum['rewards'][:n_train_accepts].to(DEVICE)
    test_rew = fly_datum['rewards'][n_train_accepts:].to(DEVICE)
    
    rule = FlyPlasticityWithW().to(DEVICE)
    opt = torch.optim.Adam(rule.parameters(), lr=lr)
    
    torch.manual_seed(seed)
    W0 = (torch.randn(10, 2) * 0.01).to(DEVICE)
    
    # Train on first portion
    for ep in range(n_epochs):
        for s in range(min(n_samp, 5)):
            inp = gen_inputs(fly_datum, seed=seed*100+s)[:split].to(DEVICE)
            opt.zero_grad()
            probs = simulate_fly(W0, inp, train_dec, train_rew, rule)
            e = 1e-7; pc = probs.clamp(e, 1-e)
            bce = -torch.mean(train_dec*torch.log(pc) + (1-train_dec)*torch.log(1-pc))
            l1 = sum(p.abs().sum() for p in rule.parameters()) * 1e-2
            loss = bce + l1
            if torch.isnan(loss) or loss.item() > 100: break
            loss.backward()
            torch.nn.utils.clip_grad_value_(rule.parameters(), 1.0)
            opt.step()
    
    # Evaluate on train and test
    rule.eval()
    with torch.no_grad():
        # Train PDE
        inp_tr = gen_inputs(fly_datum, seed=0)[:split].to(DEVICE)
        p_tr = simulate_fly(W0, inp_tr, train_dec, train_rew, rule)
        train_pde = pde(p_tr.cpu().numpy(), train_dec.cpu().numpy())
        
        # Test PDE — re-init weights for test
        W0_test = (torch.randn(10, 2) * 0.01).to(DEVICE)
        inp_te = gen_inputs(fly_datum, seed=0)[split:].to(DEVICE)
        p_te = simulate_fly(W0_test, inp_te, test_dec, test_rew, rule)
        test_pde = pde(p_te.cpu().numpy(), test_dec.cpu().numpy())
    
    w_coeff = rule.w_coeff.item()
    return train_pde, test_pde, w_coeff


def run_fig8_9():
    print("\n=== Figures 8 & 9: Held-out Validation ===")
    
    fly_data = load_fly_data()
    print(f"  {len(fly_data)} valid flies loaded")
    
    cutoffs = np.arange(0.1, 1.01, 0.05)
    
    # Storage: per fly, per cutoff
    all_train_pdes = {c: [] for c in cutoffs}
    all_test_pdes = {c: [] for c in cutoffs}
    all_w_coeffs = {c: [] for c in cutoffs}
    
    for fi, fd in enumerate(fly_data):
        print(f"  Fly {fd['fly_id']}", end='')
        for ci, cutoff in enumerate(cutoffs):
            print('.', end='', flush=True)
            tr_pde, te_pde, w_c = train_fly_split(fd, cutoff, n_epochs=200, seed=fi)
            if tr_pde is not None:
                all_train_pdes[cutoff].append(tr_pde)
                all_test_pdes[cutoff].append(te_pde)
                all_w_coeffs[cutoff].append(w_c)
        print()
    
    # Figure 8: Train/test PDE vs cutoff
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for ci, cutoff in enumerate(cutoffs):
        if cutoff >= 1.0: continue
        tr = all_train_pdes[cutoff]
        te = all_test_pdes[cutoff]
        if tr:
            ax.scatter([cutoff]*len(tr), tr, c='#43A047', alpha=0.4, s=15, zorder=3)
        if te:
            ax.scatter([cutoff]*len(te), te, c='#1565C0', alpha=0.4, s=15, zorder=3, marker='s')
    
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Trajectory Cutoff (x)')
    ax.set_ylabel('% Deviance Explained')
    ax.set_yscale('symlog', linthresh=1)
    ax.set_title('Percent Deviance Explained on Training and Test Data', fontweight='bold')
    ax.legend(['Zero line', 'Train', 'Eval'], loc='upper left')
    
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/figure8.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/figure8.png")
    
    # Figure 9: θ_001 for good vs poor test flies
    # Use cutoff=0.5 as reference
    ref_cutoff = min(cutoffs, key=lambda c: abs(c - 0.5))
    test_pdes = all_test_pdes[ref_cutoff]
    w_coeffs = all_w_coeffs[ref_cutoff]
    
    if len(test_pdes) > 2:
        good_w = [w for w, p in zip(w_coeffs, test_pdes) if p > 0]
        poor_w = [w for w, p in zip(w_coeffs, test_pdes) if p <= 0]
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        if good_w:
            # Show as boxplot across cutoffs for good flies
            axes[0].boxplot([good_w], positions=[0], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor='#A5D6A7'))
            axes[0].axhline(0, color='red', ls='--', alpha=0.5)
            axes[0].set_title(f'Good test performance ({len(good_w)} flies)', fontweight='bold')
            axes[0].set_ylabel(r'$\theta_{001}$')
        
        if poor_w:
            axes[1].boxplot([poor_w], positions=[0], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor='#EF9A9A'))
            axes[1].axhline(0, color='red', ls='--', alpha=0.5)
            axes[1].set_title(f'Poor test performance ({len(poor_w)} flies)', fontweight='bold')
            axes[1].set_ylabel(r'$\theta_{001}$')
        
        plt.tight_layout()
        plt.savefig('figures/figure9.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved figures/figure9.png")
    
    print(f"\n  Good test flies: {len(good_w) if good_w else 0}")
    print(f"  Poor test flies: {len(poor_w) if poor_w else 0}")
    if good_w: print(f"  Good θ_001 mean: {np.mean(good_w):.4f}")
    if poor_w: print(f"  Poor θ_001 mean: {np.mean(poor_w):.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Figures 8 & 9: Held-Out Validation")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    t0 = time.time()
    run_fig8_9()
    print(f"\nTotal: {time.time()-t0:.1f}s")
