"""
Tables 1 & 3: Evaluation of various reward-based plasticity rules.
Each rule is used as ground truth, then Taylor and MLP try to recover it.
Reports R² weights, R² activity, % deviance explained.

Rules are specified as lists of (coeff, x_pow, y_pow, w_pow, r_pow) tuples.
Example: x_j * r = [(1.0, 1, 0, 0, 1)]
Example: x_j * r - 0.05 * w_ij = [(1.0, 1, 0, 0, 1), (-0.05, 0, 0, 1, 0)]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import time, csv
from src.plasticity_rules import TaylorRule4Var, MLPRule
from src.network import simulate_behavior_model, behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Rule definitions ============
# Each rule: (name, [(coeff, x_pow, y_pow, w_pow, r_pow), ...])

RULES_TABLE1 = [
    ("x_j r",                        [(1.0,1,0,0,1)]),
    ("x_j r^2 - 0.05 y_i",           [(1.0,1,0,0,2), (-0.05,0,1,0,0)]),
    ("x_j r - 0.05 w_ij",            [(1.0,1,0,0,1), (-0.05,0,0,1,0)]),
    ("x_j r^2 - 0.05 x_j w_ij r",    [(1.0,1,0,0,2), (-0.05,1,0,1,1)]),
    ("x_j y_i w_ij r - 0.05 r",      [(1.0,1,1,1,1), (-0.05,0,0,0,1)]),
]

RULES_TABLE3 = [
    ("x_j y_i w_ij r - 0.05r",        [(1.0,1,1,1,1),(-0.05,0,0,0,1)]),
    ("x_j w_ij y_i r - 0.05r",        [(1.0,1,1,1,1),(-0.05,0,0,0,1)]),  # same as above
    ("x_j w_ij r^2 - 0.05r",          [(1.0,1,0,1,2),(-0.05,0,0,0,1)]),
    ("x_j y_i w_ij r^2 - 0.05r",      [(1.0,1,1,1,2),(-0.05,0,0,0,1)]),
    ("x_j r^2 - 0.05r",               [(1.0,1,0,0,2),(-0.05,0,0,0,1)]),
    ("x_j^2 y_i w_ij r - 0.05",       [(1.0,2,1,1,1),(-0.05,0,0,0,0)]),
    ("x_j^2 y_i^2 r^2 - 0.05",        [(1.0,2,2,0,2),(-0.05,0,0,0,0)]),
    ("x_j r - 0.05 x_j y_i r",        [(1.0,1,0,0,1),(-0.05,1,1,0,1)]),
    ("x_j r - 0.05 x_j w_ij r",       [(1.0,1,0,0,1),(-0.05,1,0,1,1)]),
    ("x_j r - 0.05 x_j y_i",          [(1.0,1,0,0,1),(-0.05,1,1,0,0)]),
    ("x_j r",                          [(1.0,1,0,0,1)]),
    ("x_j r - 0.05r",                  [(1.0,1,0,0,1),(-0.05,0,0,0,1)]),
    ("x_j r - 0.05 x_j y_i w_ij r",   [(1.0,1,0,0,1),(-0.05,1,1,1,1)]),
    ("x_j r - 0.05 x_j w_ij",         [(1.0,1,0,0,1),(-0.05,1,0,1,0)]),
    ("x_j r - 0.05 x_j y_i w_ij",     [(1.0,1,0,0,1),(-0.05,1,1,1,0)]),
    ("x_j r - 0.05 x_j",              [(1.0,1,0,0,1),(-0.05,1,0,0,0)]),
    ("x_j r^2 - 0.05 x_j w_ij",       [(1.0,1,0,0,2),(-0.05,1,0,1,0)]),
    ("x_j r^2 - 0.05 x_j y_i",        [(1.0,1,0,0,2),(-0.05,1,1,0,0)]),
    ("x_j r^2",                        [(1.0,1,0,0,2)]),
    ("x_j r - 0.05 w_ij",             [(1.0,1,0,0,1),(-0.05,0,0,1,0)]),
    ("x_j r^2 - 0.05 x_j y_i r",      [(1.0,1,0,0,2),(-0.05,1,1,0,1)]),
    ("x_j r^2 - 0.05 x_j y_i w_ij",   [(1.0,1,0,0,2),(-0.05,1,1,1,0)]),
    ("x_j r^2 - 0.05 x_j w_ij r",     [(1.0,1,0,0,2),(-0.05,1,0,1,1)]),
    ("x_j r^2 - 0.05 x_j y_i w_ij r", [(1.0,1,0,0,2),(-0.05,1,1,1,1)]),
    ("x_j r^2 - 0.05 x_j",            [(1.0,1,0,0,2),(-0.05,1,0,0,0)]),
    ("x_j r^2 - 0.05 x_j r",          [(1.0,1,0,0,2),(-0.05,1,0,0,1)]),
    ("x_j r - 0.05 y_i w_ij r",       [(1.0,1,0,0,1),(-0.05,0,1,1,1)]),
    ("x_j r - 0.05 y_i r",            [(1.0,1,0,0,1),(-0.05,0,1,0,1)]),
    ("x_j r - 0.05 w_ij r",           [(1.0,1,0,0,1),(-0.05,0,0,1,1)]),
    ("x_j r - 0.05 y_i w_ij",         [(1.0,1,0,0,1),(-0.05,0,1,1,0)]),
    ("y_i w_ij r^2 - 0.05",           [(1.0,0,1,1,2),(-0.05,0,0,0,0)]),
    ("x_j r^2 - 0.05 w_ij",           [(1.0,1,0,0,2),(-0.05,0,0,1,0)]),
    ("x_j r^2 - 0.05 y_i w_ij r",     [(1.0,1,0,0,2),(-0.05,0,1,1,1)]),
    ("x_j r^2 - 0.05 y_i r",          [(1.0,1,0,0,2),(-0.05,0,1,0,1)]),
    ("x_j r^2 - 0.05 w_ij",           [(1.0,1,0,0,2),(-0.05,0,0,1,0)]),
    ("x_j^2 y_i w_ij r^2 - 0.05r",    [(1.0,2,1,1,2),(-0.05,0,0,0,1)]),
    ("x_j r^2 - 0.05 y_i r",          [(1.0,1,0,0,2),(-0.05,0,1,0,1)]),
    ("y_i w_ij r - 0.05",             [(1.0,0,1,1,1),(-0.05,0,0,0,0)]),
    ("y_i^2 r^2 - 0.05",              [(1.0,0,2,0,2),(-0.05,0,0,0,0)]),
    ("x_j r^2 - 0.05 y_i",            [(1.0,1,0,0,2),(-0.05,0,1,0,0)]),
    ("x_j^2 y_i^2 r^2 - 0.05r",       [(1.0,2,2,0,2),(-0.05,0,0,0,1)]),
    ("x_j^2 y_i w_ij r - 0.05r",      [(1.0,2,1,1,1),(-0.05,0,0,0,1)]),
    ("y_i w_ij r - 0.05 x_j r",       [(1.0,0,1,1,1),(-0.05,1,0,0,1)]),
    ("x_j^2 y_i^2 r - 0.05r",         [(1.0,2,2,0,1),(-0.05,0,0,0,1)]),
    ("y_i w_ij r^2 - 0.05 x_j r",     [(1.0,0,1,1,2),(-0.05,1,0,0,1)]),
    ("y_i w_ij r^2 - 0.05r",          [(1.0,0,1,1,2),(-0.05,0,0,0,1)]),
    ("y_i^2 r^2 - 0.05r",             [(1.0,0,2,0,2),(-0.05,0,0,0,1)]),
]

# Remove duplicates by name
seen = set()
RULES_TABLE3_UNIQUE = []
for name, terms in RULES_TABLE3:
    if name not in seen:
        seen.add(name)
        RULES_TABLE3_UNIQUE.append((name, terms))
RULES_TABLE3 = RULES_TABLE3_UNIQUE


def compute_dw_gt(x, y, w, r, rule_terms, n_hidden, n_input):
    """Compute ground truth dW from rule specification.
    rule_terms: list of (coeff, x_pow, y_pow, w_pow, r_pow)
    """
    dW = torch.zeros(n_hidden, n_input)
    x_exp = x.unsqueeze(0).expand(n_hidden, -1)
    y_exp = y.unsqueeze(1).expand(-1, n_input)
    
    for coeff, xp, yp, wp, rp in rule_terms:
        term = coeff * (x_exp ** xp) * (y_exp ** yp) * (w ** wp) * (r ** rp)
        dW = dW + term
    return dW


def gen_gt_general(seed, rule_terms, n_input=2, n_hidden=10, traj_len=240,
                   firing_mean=0.75, noise_var=0.015, ma_win=10):
    """Generate ground truth with arbitrary plasticity rule."""
    torch.manual_seed(seed); np.random.seed(seed)
    rp = np.array([[0.2,0.8],[0.9,0.1],[0.2,0.8]])
    bs = traj_len // 3
    rs = np.concatenate([np.tile(rp[b], (bs,1)) for b in range(3)])
    
    W = torch.randn(n_hidden, n_input) * 0.01; W0 = W.clone()
    lr = 1.0 / n_input
    
    inps, chs, rews, wts, hids = [], [], [], [], []
    ravg = 0.0
    
    for t in range(traj_len):
        oi = int(torch.randint(0, 2, (1,)).item())
        x = torch.zeros(n_input); x[oi] = firing_mean
        x = x + torch.randn(n_input) * (noise_var ** 0.5)
        inps.append(x)
        
        h, _, p = behavior_forward(W, x, n_hidden)
        hids.append(h.clone()); wts.append(W.clone())
        
        c = int(torch.bernoulli(torch.tensor(p.item())).item()); chs.append(c)
        R = 0
        if c == 1: R = int(np.random.binomial(1, rs[t, oi]))
        rews.append(float(R))
        
        r_val = R - ravg  # reward - expected reward
        a = 1.0 / ma_win; ravg = (1-a) * ravg + a * R
        
        # Apply ground truth rule
        dW = compute_dw_gt(x, h, W, r_val, rule_terms, n_hidden, n_input)
        W = W + lr * dW
        W = W.clamp(-10.0, 10.0)
    
    return {'inputs': torch.stack(inps), 'choices': torch.tensor(chs, dtype=torch.float32),
            'rewards': torch.tensor(rews, dtype=torch.float32),
            'weights': torch.stack(wts), 'hidden': torch.stack(hids), 'W0': W0}


def r2(yt, yp):
    sr = np.sum((yt-yp)**2); st = np.sum((yt-yt.mean())**2)
    return 1.0 - sr / (st + 1e-10)

def pde(probs, choices):
    e = 1e-7; p = np.clip(probs, e, 1-e); c = np.array(choices)
    llm = np.sum(c*np.log(p)+(1-c)*np.log(1-p))
    pn = np.clip(c.mean(), e, 1-e)
    lln = np.sum(c*np.log(pn)+(1-c)*np.log(1-pn))
    return 100.0 * (1.0 - (-2*llm) / (-2*lln + 1e-10))


def train_eval_rule(rule_terms, model_type='taylor', n_epochs=400, n_seeds=3):
    """Train and evaluate one rule with one model type, averaged over seeds."""
    r2ws, r2as, pds = [], [], []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000)
        
        # Generate data with this ground truth rule
        data = [gen_gt_general(seed*100+i, rule_terms) for i in range(25)]
        train_data, eval_data = data[:18], data[18:]
        
        # Create model
        if model_type == 'taylor':
            rule = TaylorRule4Var(init_scale=1e-5).to(DEVICE)
        else:
            rule = MLPRule(input_dim=4, hidden_dim=10).to(DEVICE)
        
        optimizer = torch.optim.Adam(rule.parameters(), lr=1e-3)
        nt = len(train_data)
        
        for ep in range(n_epochs):
            d = train_data[ep % nt]
            inp = d['inputs'].to(DEVICE); rew = d['rewards'].to(DEVICE)
            cho = d['choices'].to(DEVICE); W0 = d['W0'].to(DEVICE)
            
            optimizer.zero_grad()
            probs, wl, hiddens = simulate_behavior_model(W0, inp, rew, rule)
            
            eps = 1e-7; pc = probs.clamp(eps, 1-eps)
            bce = -torch.mean(cho*torch.log(pc) + (1-cho)*torch.log(1-pc))
            loss = bce
            if model_type == 'taylor':
                loss = loss + 1e-2 * rule.coeffs.abs().sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(rule.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate on eval data
        rule.eval()
        seed_r2w, seed_r2a, seed_pde = [], [], []
        with torch.no_grad():
            for d in eval_data:
                inp = d['inputs'].to(DEVICE); rew = d['rewards'].to(DEVICE)
                W0 = d['W0'].to(DEVICE)
                probs, wl, hiddens = simulate_behavior_model(W0, inp, rew, rule)
                
                mw = torch.stack(wl).cpu().numpy(); gw = d['weights'].numpy()
                mh = hiddens.cpu().numpy(); gh = d['hidden'].numpy()
                
                seed_r2w.append(r2(gw.flatten(), mw.flatten()))
                seed_r2a.append(r2(gh.flatten(), mh.flatten()))
                seed_pde.append(pde(probs.cpu().numpy(), d['choices'].numpy()))
        rule.train()
        
        r2ws.append(np.median(seed_r2w))
        r2as.append(np.median(seed_r2a))
        pds.append(np.median(seed_pde))
    
    return np.mean(r2ws), np.mean(r2as), np.mean(pds)


if __name__ == '__main__':
    print("=" * 70)
    print("Tables 1 & 3: Evaluation of Plasticity Rules")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    t0 = time.time()
    
    # Combine all rules (Table 1 is subset of Table 3)
    all_rules = []
    seen_names = set()
    for name, terms in RULES_TABLE1 + RULES_TABLE3:
        if name not in seen_names:
            seen_names.add(name)
            all_rules.append((name, terms))
    
    print(f"\nTotal unique rules to test: {len(all_rules)}")
    print(f"Each rule: 2 models × 3 seeds × 400 epochs = ~{len(all_rules)*2} training runs\n")
    
    results = []
    
    for idx, (name, terms) in enumerate(all_rules):
        print(f"[{idx+1}/{len(all_rules)}] Rule: {name}")
        
        try:
            # Taylor
            t_r2w, t_r2a, t_pde = train_eval_rule(terms, 'taylor', n_epochs=400, n_seeds=3)
            print(f"  Taylor: R2_w={t_r2w:.2f} R2_a={t_r2a:.2f} %Dev={t_pde:.2f}")
        except Exception as e:
            print(f"  Taylor FAILED: {e}")
            t_r2w, t_r2a, t_pde = 0, 0, 0
        
        try:
            # MLP
            m_r2w, m_r2a, m_pde = train_eval_rule(terms, 'mlp', n_epochs=400, n_seeds=3)
            print(f"  MLP:    R2_w={m_r2w:.2f} R2_a={m_r2a:.2f} %Dev={m_pde:.2f}")
        except Exception as e:
            print(f"  MLP FAILED: {e}")
            m_r2w, m_r2a, m_pde = 0, 0, 0
        
        results.append({
            'rule': name,
            'mlp_r2w': m_r2w, 'mlp_r2a': m_r2a, 'mlp_pde': m_pde,
            'taylor_r2w': t_r2w, 'taylor_r2a': t_r2a, 'taylor_pde': t_pde,
        })
        
        # Save intermediate results
        with open('results/tables1_3.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['rule','mlp_r2w','mlp_r2a','mlp_pde',
                                               'taylor_r2w','taylor_r2a','taylor_pde'])
            w.writeheader()
            w.writerows(results)
    
    # Print final table
    print("\n" + "=" * 100)
    print("COMPLETE RESULTS — Tables 1 & 3")
    print("=" * 100)
    print(f"{'Plasticity Rule':<35} {'MLP':^30} {'Taylor':^30}")
    print(f"{'':>35} {'R2_W':>8} {'R2_A':>8} {'%Dev':>8} {'R2_W':>10} {'R2_A':>8} {'%Dev':>8}")
    print("-" * 100)
    
    # Sort by Taylor %Dev (descending) to match paper
    results_sorted = sorted(results, key=lambda r: r['taylor_pde'], reverse=True)
    for r in results_sorted:
        print(f"{r['rule']:<35} {r['mlp_r2w']:>8.2f} {r['mlp_r2a']:>8.2f} {r['mlp_pde']:>8.2f}"
              f" {r['taylor_r2w']:>10.2f} {r['taylor_r2a']:>8.2f} {r['taylor_pde']:>8.2f}")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s ({(time.time()-t0)/3600:.1f} hours)")
    print(f"Results saved to results/tables1_3.csv")
