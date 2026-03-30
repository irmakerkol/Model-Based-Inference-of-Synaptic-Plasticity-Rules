"""
Experiment 3: Inferring plasticity in the fruit fly (Drosophila).
PyTorch — reproduces Section 5 and Figure 4.

Uses real experimental data from Rajagopalan et al. (2023).
Handles variable-length trials: Y contains both 0 (reject) and 1 (accept).
Plasticity update only occurs on accept trials (when reward is received).

Paper's 4 model comparison:
  1. With w_ij + R-E[R]
  2. Without w_ij + R-E[R]  
  3. With w_ij + raw R
  4. Without w_ij + raw R
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import time

from src.plasticity_rules import FlyPlasticityWithW, FlyPlasticityWithoutW
from src.network import behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ Data loading ============

def load_fly_data(data_dir='./data/', n_flies=18):
    """Load processed fly .mat files with variable-length trial support."""
    fly_data = []
    for i in range(1, n_flies + 1):
        path = os.path.join(data_dir, f'Fly{i}.mat')
        if not os.path.exists(path):
            continue
        d = sio.loadmat(path)
        X = d['X']                    # (n_timesteps, 2)
        Y = d['Y'].flatten()          # (n_timesteps,) 0=reject, 1=accept
        R = d['R'].flatten()          # (n_trials,) one per accept
        
        fly_data.append({
            'odors': torch.tensor(X, dtype=torch.float32),
            'decisions': torch.tensor(Y, dtype=torch.float32),
            'rewards': torch.tensor(R, dtype=torch.float32),
            'n_timesteps': len(Y),
            'n_trials': int(Y.sum()),
            'fly_id': i,
        })
    
    print(f"  Loaded {len(fly_data)} flies from {data_dir}")
    for d in fly_data:
        n_rej = d['n_timesteps'] - d['n_trials']
        rej_pct = 100.0 * n_rej / d['n_timesteps'] if d['n_timesteps'] > 0 else 0
        flag = "" if n_rej > 10 else " ← FEW REJECTS"
        print(f"    Fly {d['fly_id']:2d}: {d['n_timesteps']:4d} steps, "
              f"{d['n_trials']:3d} trials, {n_rej:3d} rejects ({rej_pct:.0f}%){flag}")
    return fly_data


def generate_inputs_for_fly(fly_datum, input_firing_mean=0.75,
                             input_noise_var=0.015, seed=0):
    """Generate neural input encodings from odor identity."""
    torch.manual_seed(seed)
    odors = fly_datum['odors']  # (n_timesteps, 2) one-hot
    n = odors.shape[0]
    noise = torch.randn(n, 2) * (input_noise_var ** 0.5)
    return odors * input_firing_mean + noise


# ============ Simulation with variable-length trials ============

def simulate_fly_behavior(W0, inputs, decisions, rewards, plasticity_rule,
                           moving_avg_window=10, use_reward_expectation=True,
                           n_hidden=10):
    """Simulate fly behavior with variable-length trials.
    
    Key difference from Experiment 2: plasticity update ONLY happens
    when the fly accepts (decision=1), not on reject timesteps.
    
    The model produces a probability at each timestep.
    The loss is computed over ALL timesteps (both accepts and rejects).
    But the plasticity rule is applied only after accepts.
    
    Args:
        W0: (n_hidden, n_input) initial weights
        inputs: (n_timesteps, n_input) 
        decisions: (n_timesteps,) 0=reject, 1=accept
        rewards: (n_trials,) one per accept
        plasticity_rule: FlyPlasticityWithW or WithoutW
        
    Returns:
        all_probs: (n_timesteps,) model predicted probabilities
    """
    n_timesteps = inputs.shape[0]
    n_input = inputs.shape[1]
    W = W0.clone()
    
    lr = 1.0 / n_input  # Paper: 1/input_dim
    reward_avg = torch.tensor(0.0, device=W0.device, dtype=W0.dtype)
    alpha = 1.0 / moving_avg_window
    
    all_probs = []
    trial_idx = 0  # Index into rewards array
    
    for t in range(n_timesteps):
        x = inputs[t]
        
        # Forward pass: tanh hidden + fixed output (matching paper)
        hidden, logits, prob = behavior_forward(W, x, n_hidden)
        all_probs.append(prob)
        
        # Plasticity update ONLY on accept trials
        if decisions[t] > 0.5:  # Accept
            if trial_idx < len(rewards):
                R = rewards[trial_idx]
                trial_idx += 1
            else:
                R = torch.tensor(0.0, device=W0.device)
            
            if use_reward_expectation:
                r = R - reward_avg
            else:
                r = R
            
            reward_avg = ((1 - alpha) * reward_avg + alpha * R).detach()
            
            # Apply plasticity
            dW = plasticity_rule(x, hidden, W, r)
            dW = dW.clamp(-1.0, 1.0)  # cap single update magnitude
            W = W + lr * dW
            W = W.clamp(-5.0, 5.0)  # tighter clamp for real data stability
    
    return torch.stack(all_probs)


# ============ Metrics ============

def percent_deviance_explained(probs, decisions):
    """Compute %Dev using ALL timesteps (accepts and rejects)."""
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    decisions = np.array(decisions)
    
    ll_model = np.sum(decisions * np.log(probs) + 
                      (1 - decisions) * np.log(1 - probs))
    p_null = np.clip(decisions.mean(), eps, 1 - eps)
    ll_null = np.sum(decisions * np.log(p_null) + 
                     (1 - decisions) * np.log(1 - p_null))
    
    dev_model = -2 * ll_model
    dev_null = -2 * ll_null
    return 100.0 * (1.0 - dev_model / (dev_null + 1e-10))


# ============ Training ============

def train_single_fly(fly_datum, with_weight_term=True, use_reward_expectation=True,
                     n_epochs=250, lr=1e-3, l1_reg=1e-2,
                     n_hidden=10, n_input=2, num_samplings=20, seed=0):
    """Train plasticity model for one fly."""
    
    if with_weight_term:
        rule = FlyPlasticityWithW().to(DEVICE)
    else:
        rule = FlyPlasticityWithoutW().to(DEVICE)
    
    optimizer = torch.optim.Adam(rule.parameters(), lr=lr)
    
    # Generate multiple input resamplings
    all_inputs = []
    for s in range(num_samplings):
        inp = generate_inputs_for_fly(fly_datum, seed=seed * 1000 + s)
        all_inputs.append(inp.to(DEVICE))
    
    decisions = fly_datum['decisions'].to(DEVICE)
    rewards = fly_datum['rewards'].to(DEVICE)
    
    # Random initial weights (paper: scale=0.01)
    torch.manual_seed(seed)
    W0 = (torch.randn(n_hidden, n_input) * 0.01).to(DEVICE)
    
    diverged = False
    for epoch in range(n_epochs):
        if diverged:
            break
        for s_idx in range(num_samplings):
            inp = all_inputs[s_idx]
            optimizer.zero_grad()
            
            probs = simulate_fly_behavior(
                W0, inp, decisions, rewards, rule,
                moving_avg_window=10,
                use_reward_expectation=use_reward_expectation,
                n_hidden=n_hidden
            )
            
            # BCE loss over ALL timesteps
            eps = 1e-7
            probs_c = probs.clamp(eps, 1 - eps)
            bce = -torch.mean(decisions * torch.log(probs_c) +
                             (1 - decisions) * torch.log(1 - probs_c))
            
            l1 = sum(p.abs().sum() for p in rule.parameters()) * l1_reg
            loss = bce + l1
            
            if torch.isnan(loss) or loss.item() > 100:
                diverged = True
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(rule.parameters(), 1.0)
            optimizer.step()
    
    # Evaluate: median %Dev across samplings
    rule.eval()
    pde_list = []
    with torch.no_grad():
        for s_idx in range(num_samplings):
            probs = simulate_fly_behavior(
                W0, all_inputs[s_idx], decisions, rewards, rule,
                moving_avg_window=10,
                use_reward_expectation=use_reward_expectation,
                n_hidden=n_hidden
            )
            pde_list.append(percent_deviance_explained(
                probs.cpu().numpy(), fly_datum['decisions'].numpy()
            ))
    rule.train()
    
    pde = np.median(pde_list)
    
    params = {}
    if with_weight_term:
        params = {'bias': rule.bias.item(), 'w': rule.w_coeff.item(),
                  'x': rule.x_coeff.item(), 'r': rule.r_coeff.item(),
                  'xr': rule.xr_coeff.item()}
    else:
        params = {'bias': rule.bias.item(), 'x': rule.x_coeff.item(),
                  'r': rule.r_coeff.item(), 'xr': rule.xr_coeff.item()}
    
    return {'params': params, 'pde': pde, 'final_loss': loss.item()}


def train_all_flies(fly_data, with_weight_term=True, use_reward_expectation=True,
                    n_epochs=250, lr=1e-3, num_samplings=20, verbose=True):
    results = []
    for fly_idx, d in enumerate(fly_data):
        result = train_single_fly(
            d, with_weight_term=with_weight_term,
            use_reward_expectation=use_reward_expectation,
            n_epochs=n_epochs, lr=lr, num_samplings=num_samplings,
            seed=fly_idx
        )
        results.append(result)
        if verbose:
            w_str = f", w={result['params'].get('w', 0):.4f}" if with_weight_term else ""
            print(f"    Fly {d['fly_id']:2d} | PDE: {result['pde']:6.2f}% | "
                  f"xr={result['params']['xr']:.4f}{w_str}")
    return results


# ============ Plotting ============

def plot_figure4(res_with_w, res_without_w, res_exp, res_raw,
                 save_path='figures/figure4.png'):
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    n_flies = len(res_with_w)
    
    term_names = ['bias', 'w', 'x', 'r', 'xr']
    term_labels = ['bias', r'$w_{ij}$', r'$x_j$', r'$r$', r'$x_j r$']
    x_pos = np.arange(len(term_names))
    width = 0.35
    
    # Panel C: coefficients
    ax_c = fig.add_subplot(gs[0, 0:2])
    wv = {t: [r['params'].get(t, 0) for r in res_with_w] for t in term_names}
    wo_names = ['bias', 'x', 'r', 'xr']
    wov = {t: [r['params'][t] for r in res_without_w] for t in wo_names}
    bp1 = ax_c.boxplot([wv[t] for t in term_names],
                        positions=x_pos - width/2, widths=width*0.8, patch_artist=True)
    for p in bp1['boxes']: p.set_facecolor('lightblue')
    bp2 = ax_c.boxplot([wov[t] for t in wo_names],
                        positions=[p + width/2 for p in [0, 2, 3, 4]],
                        widths=width*0.8, patch_artist=True)
    for p in bp2['boxes']: p.set_facecolor('lightsalmon')
    ax_c.set_xticks(x_pos); ax_c.set_xticklabels(term_labels)
    ax_c.set_ylabel(r'$\theta$ Value'); ax_c.set_title('C. Inferred Coefficients')
    ax_c.axhline(0, color='k', ls='--', lw=0.5)
    ax_c.legend([bp1['boxes'][0], bp2['boxes'][0]],
                ['With $w_{ij}$', 'Without $w_{ij}$'], fontsize=8)
    
    # Panel D: %Dev with vs without
    ax_d = fig.add_subplot(gs[0, 2])
    pw = [r['pde'] for r in res_with_w]
    pwo = [r['pde'] for r in res_without_w]
    bp3 = ax_d.boxplot([pwo, pw], tick_labels=['Without\n$w_{ij}$', 'With\n$w_{ij}$'],
                        patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightsalmon'); bp3['boxes'][1].set_facecolor('lightblue')
    ax_d.set_ylabel('% Deviance Explained'); ax_d.set_title('D. Goodness of Fit')
    s1, p1 = stats.wilcoxon(pw, pwo)
    ax_d.text(1.5, max(max(pw), max(pwo)) * 1.05, f'p = {p1:.2e}',
              ha='center', fontsize=9, color='red')
    
    # Panel E: expected vs raw
    ax_e = fig.add_subplot(gs[1, 0:2])
    ev = {t: [r['params'].get(t, 0) for r in res_exp] for t in term_names}
    rv = {t: [r['params'].get(t, 0) for r in res_raw] for t in term_names}
    bp4 = ax_e.boxplot([ev[t] for t in term_names],
                        positions=x_pos - width/2, widths=width*0.8, patch_artist=True)
    for p in bp4['boxes']: p.set_facecolor('lightblue')
    bp5 = ax_e.boxplot([rv[t] for t in term_names],
                        positions=x_pos + width/2, widths=width*0.8, patch_artist=True)
    for p in bp5['boxes']: p.set_facecolor('lightsalmon')
    ax_e.set_xticks(x_pos); ax_e.set_xticklabels(term_labels)
    ax_e.set_ylabel(r'$\theta$ Value'); ax_e.set_title('E. Expected vs Raw Reward')
    ax_e.axhline(0, color='k', ls='--', lw=0.5)
    ax_e.legend([bp4['boxes'][0], bp5['boxes'][0]], ['R-E[R]', 'R (raw)'], fontsize=8)
    
    # Panel F
    ax_f = fig.add_subplot(gs[1, 2])
    pe = [r['pde'] for r in res_exp]
    pr = [r['pde'] for r in res_raw]
    bp6 = ax_f.boxplot([pe, pr], tick_labels=['R-E[R]', 'R (raw)'], patch_artist=True)
    bp6['boxes'][0].set_facecolor('lightblue'); bp6['boxes'][1].set_facecolor('lightsalmon')
    ax_f.set_ylabel('% Deviance Explained'); ax_f.set_title('F. Expected vs Raw Reward')
    s2, p2 = stats.wilcoxon(pe, pr)
    ax_f.text(1.5, max(max(pe), max(pr)) * 1.05, f'p = {p2:.3f}',
              ha='center', fontsize=9, color='red')
    
    plt.suptitle("Figure 4: Inferring Principles of Plasticity in the Fruit Fly\n(Real Data — 18 Flies)",
                 fontsize=14, fontweight='bold', y=0.99)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============ Main ============

if __name__ == '__main__':
    print("=" * 70)
    print("EXPERIMENT 3: Fruit Fly Plasticity — Real Data (PyTorch)")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    
    t0 = time.time()
    
    print("\n[0/4] Loading fly data...")
    fly_data = load_fly_data('./data/', n_flies=18)
    if not fly_data:
        print("No data! Run process_fly_data_v2.py first.")
        sys.exit(1)
    
    NE = 250; NS = 20; LR = 1e-3
    
    print(f"\n[1/4] WITH w_ij + R-E[R]...")
    r1 = train_all_flies(fly_data, True, True, NE, LR, NS)
    
    print(f"\n[2/4] WITHOUT w_ij + R-E[R]...")
    r2 = train_all_flies(fly_data, False, True, NE, LR, NS)
    
    print(f"\n[3/4] WITH w_ij + raw R...")
    r3 = train_all_flies(fly_data, True, False, NE, LR, NS)
    
    # Filter out diverged flies (0 rejects = accept-only, causes explosion)
    valid_mask = [d['n_timesteps'] > d['n_trials'] for d in fly_data]  # has rejects
    n_valid = sum(valid_mask)
    n_total = len(fly_data)
    
    r1_valid = [r for r, v in zip(r1, valid_mask) if v]
    r2_valid = [r for r, v in zip(r2, valid_mask) if v]
    r3_valid = [r for r, v in zip(r3, valid_mask) if v]
    
    print(f"\n{'='*60}")
    print(f"RESULTS ({n_valid}/{n_total} flies with reject data)")
    print(f"{'='*60}")
    
    pw = [r['pde'] for r in r1_valid]
    pwo = [r['pde'] for r in r2_valid]
    pr = [r['pde'] for r in r3_valid]
    
    _, p1 = stats.wilcoxon(pw, pwo)
    _, p2 = stats.wilcoxon(pw, pr)
    wv = [r['params']['w'] for r in r1_valid]
    
    print(f"\nWith vs Without w_ij:")
    print(f"  With:    {np.mean(pw):.1f}% ± {np.std(pw):.1f}%")
    print(f"  Without: {np.mean(pwo):.1f}% ± {np.std(pwo):.1f}%")
    print(f"  Wilcoxon p = {p1:.4e}  (paper: p=5e-5)")
    
    print(f"\nθ_w (weight term) per fly:")
    for r, v in zip(r1, valid_mask):
        if v:
            print(f"  {r['params']['w']:+.4f}", end="")
    print(f"\n  Mean = {np.mean(wv):.4f}  (paper: negative = forgetting)")
    
    print(f"\nR-E[R] vs raw R:")
    print(f"  R-E[R]: {np.mean(pw):.1f}%")
    print(f"  R:      {np.mean(pr):.1f}%")
    print(f"  Wilcoxon p = {p2:.4f}  (paper: p=0.067)")
    
    # Plot only valid flies
    plot_figure4(r1_valid, r2_valid, r1_valid, r3_valid,
                 save_path='figures/figure4.png')
    print(f"\nDone in {time.time()-t0:.1f}s")
