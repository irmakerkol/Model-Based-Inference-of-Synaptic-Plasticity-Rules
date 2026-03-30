"""
Experiment 2: Recovery of reward-based plasticity rule from simulated behavior.
PyTorch implementation — reproduces Section 4 and Figure 3.

ALL settings verified against paper's source code (run.py, model.py, data_loader.py):
  - Architecture: [2, 10, 1] with tanh hidden, fixed output (5.0/n_hidden)
  - Weight init: Gaussian(0, 0.01) for plastic layer
  - Ground truth: dw_ij = x_j * (R - E[R]), applied with lr = 1/input_dim = 0.5
  - input_firing_mean = 0.75, input_variance = 0.015 (NOT 0.05!)
  - Moving average window = 10, initial E[R] = 0 (history starts as zeros)
  - BCE loss on logits (optax.sigmoid_binary_cross_entropy)
  - L1 regularization = 1e-2 on Taylor coefficients
  - Taylor init: scale = 1e-5 (paper's synapse.py init_random)
  - MLP: leaky_relu hidden + tanh output, init scale = 0.01
  - 18 train + 7 eval trajectories per seed, 3 seeds, median then mean
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from src.plasticity_rules import TaylorRule4Var, MLPRule
from src.network import simulate_behavior_model, behavior_forward

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ Ground truth data generation ============

def generate_ground_truth_behavior(seed, n_input=2, n_hidden=10,
                                    trajectory_length=240,
                                    input_firing_mean=0.75,
                                    input_noise_var=0.015,  # Paper: input_variance=0.015
                                    moving_avg_window=10):
    """Generate one behavioral trajectory matching paper's data_loader.py exactly.
    
    Key differences from previous version:
    - input_variance = 0.015 (paper's run.py config)
    - hidden = tanh(W @ x), not sigmoid
    - output = sigmoid(5.0/n_hidden * sum(hidden)), not mean(sigmoid)
    - weight init scale = 0.01 (paper's model.py initialize_params)
    - lr = 1/input_dim for weight updates (paper's model.py update_params)
    - initial E[R] = 0 (paper: r_history starts as all zeros)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    block_size = trajectory_length // 3
    reward_probs = np.array([[0.2, 0.8], [0.9, 0.1], [0.2, 0.8]])
    reward_schedule = np.concatenate([
        np.tile(reward_probs[0], (block_size, 1)),
        np.tile(reward_probs[1], (block_size, 1)),
        np.tile(reward_probs[2], (block_size, 1)),
    ])
    
    # Paper: initialize_params with scale=0.01 (not Kaiming)
    W = torch.randn(n_hidden, n_input) * 0.01
    W0 = W.clone()
    
    # Paper: lr = 1.0 / input_dim
    lr = 1.0 / n_input
    
    inputs, choices, rewards = [], [], []
    weights_list, hidden_list, probs_list = [], [], []
    
    # Paper: r_history = deque(moving_avg_window * [0], ...) → initial E[R] = 0
    reward_avg = 0.0
    
    for t in range(trajectory_length):
        odor_idx = int(torch.randint(0, 2, (1,)).item())
        
        # Paper: x = noise + mus[odor], sigma = sqrt(input_variance)
        x = torch.zeros(n_input)
        x[odor_idx] = input_firing_mean
        x = x + torch.randn(n_input) * (input_noise_var ** 0.5)
        inputs.append(x)
        
        # Forward: tanh hidden + fixed output (paper's network_forward)
        hidden, logits, prob = behavior_forward(W, x, n_hidden)
        
        hidden_list.append(hidden.clone())
        probs_list.append(prob.item())
        weights_list.append(W.clone())
        
        # Stochastic choice
        choice = int(torch.bernoulli(torch.tensor(prob.item())).item())
        choices.append(choice)
        
        # Reward only if accepted
        R = 0
        if choice == 1:
            R = int(np.random.binomial(1, reward_schedule[t, odor_idx]))
        rewards.append(float(R))
        
        # Paper: reward_term = reward - expected_reward
        r = R - reward_avg
        
        # Update running average (paper uses deque, equivalent to exponential)
        alpha = 1.0 / moving_avg_window
        reward_avg = (1 - alpha) * reward_avg + alpha * R
        
        # Ground truth covariance rule: dw_ij = x_j * r (same for all hidden neurons)
        # Paper: vmap_synapses computes dw per synapse, then applies lr
        dW = x.unsqueeze(0).expand(n_hidden, -1) * r
        W = W + lr * dW  # Paper: w + lr * dw, lr = 1/input_dim
    
    return {
        'inputs': torch.stack(inputs),
        'choices': torch.tensor(choices, dtype=torch.float32),
        'rewards': torch.tensor(rewards, dtype=torch.float32),
        'weights': torch.stack(weights_list),
        'hidden': torch.stack(hidden_list),
        'output_probs': torch.tensor(probs_list),
        'W0': W0,
    }


# ============ Metrics ============

def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def percent_deviance_explained(probs, choices):
    """Percent deviance explained (Equation 11)."""
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    choices = np.array(choices)
    ll_model = np.sum(choices * np.log(probs) + (1 - choices) * np.log(1 - probs))
    p_null = np.clip(choices.mean(), eps, 1 - eps)
    ll_null = np.sum(choices * np.log(p_null) + (1 - choices) * np.log(1 - p_null))
    return 100.0 * (1.0 - (-2 * ll_model) / (-2 * ll_null + 1e-10))


# ============ Training (single seed) ============

def train_single_seed(train_data, eval_data, model_type='taylor',
                      n_epochs=400, lr=1e-3, l1_reg=1e-2, seed=0, verbose=True):
    """Train on one seed. 18 train + 7 eval trajectories."""
    torch.manual_seed(seed)
    
    if model_type == 'taylor':
        # Paper: init_random uses scale=1e-5
        rule = TaylorRule4Var(init_scale=1e-5).to(DEVICE)
    elif model_type == 'mlp':
        # Paper: init_plasticity_mlp with scale=0.01, leaky_relu+tanh
        rule = MLPRule(input_dim=4, hidden_dim=10).to(DEVICE)
    else:
        raise ValueError(f"Unknown: {model_type}")
    
    optimizer = torch.optim.Adam(rule.parameters(), lr=lr)
    n_train = len(train_data)
    loss_history = []
    coeff_history = []
    
    for epoch in range(n_epochs):
        d = train_data[epoch % n_train]
        inp = d['inputs'].to(DEVICE)
        rew = d['rewards'].to(DEVICE)
        cho = d['choices'].to(DEVICE)
        W0 = d['W0'].to(DEVICE)
        
        optimizer.zero_grad()
        
        probs, wt_list, hiddens = simulate_behavior_model(
            W0, inp, rew, rule, moving_avg_window=10, use_reward_expectation=True
        )
        
        # BCE loss (paper uses optax.sigmoid_binary_cross_entropy)
        eps = 1e-7
        probs_c = probs.clamp(eps, 1 - eps)
        bce = -torch.mean(cho * torch.log(probs_c) + (1 - cho) * torch.log(1 - probs_c))
        
        # L1 on Taylor only (paper: regularization_scale=1e-2, "We do not apply L1 to MLP")
        loss = bce
        if model_type == 'taylor' and l1_reg > 0:
            loss = loss + l1_reg * rule.coeffs.abs().sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(rule.parameters(), 1.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        if model_type == 'taylor':
            coeff_history.append(rule.coeffs.detach().cpu().numpy().copy())
        
        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            if model_type == 'taylor':
                c = rule.coeffs.detach().cpu()
                print(f"    Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                      f"theta_1001(x*r): {c[1,0,0,1]:.4f} (target~1)")
            else:
                print(f"    Epoch {epoch:4d} | Loss: {loss.item():.4f}")
    
    def evaluate_on_data(data_list):
        results = []
        rule.eval()
        with torch.no_grad():
            for d in data_list:
                inp = d['inputs'].to(DEVICE)
                rew = d['rewards'].to(DEVICE)
                W0 = d['W0'].to(DEVICE)
                
                probs, wt_list, hiddens = simulate_behavior_model(
                    W0, inp, rew, rule, moving_avg_window=10, use_reward_expectation=True
                )
                
                probs_np = probs.cpu().numpy()
                choices_np = d['choices'].numpy()
                gt_w = d['weights'].numpy()
                gt_h = d['hidden'].numpy()
                model_w = torch.stack(wt_list).cpu().numpy()
                model_h = hiddens.cpu().numpy()
                
                results.append({
                    'r2_weights': r2_score_np(gt_w.flatten(), model_w.flatten()),
                    'r2_activity': r2_score_np(gt_h.flatten(), model_h.flatten()),
                    'pde': percent_deviance_explained(probs_np, choices_np),
                    'probs': probs_np,
                    'model_weights': model_w,
                })
        rule.train()
        return results
    
    train_results = evaluate_on_data(train_data)
    eval_results = evaluate_on_data(eval_data)
    
    return {
        'params': rule,
        'loss_history': loss_history,
        'coeff_history': coeff_history,
        'train_results': train_results,
        'eval_results': eval_results,
        'model_type': model_type,
    }


# ============ Multi-seed training ============

def train_behavior_model(model_type='taylor', n_epochs=400, lr=1e-3,
                          n_seeds=3, verbose=True):
    """Train with multiple seeds. Median per seed, average over seeds."""
    all_seed_results = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n  --- Seed {seed+1}/{n_seeds} ---")
        
        all_data = [generate_ground_truth_behavior(seed=seed * 100 + i) for i in range(25)]
        train_data = all_data[:18]
        eval_data = all_data[18:25]
        
        result = train_single_seed(
            train_data, eval_data, model_type=model_type,
            n_epochs=n_epochs, lr=lr, seed=seed * 1000, verbose=verbose
        )
        
        med = {
            'r2_w': np.median([r['r2_weights'] for r in result['train_results']]),
            'r2_a': np.median([r['r2_activity'] for r in result['train_results']]),
            'pde':  np.median([r['pde'] for r in result['train_results']]),
        }
        
        if verbose:
            print(f"    Seed {seed+1} median: R2_w={med['r2_w']:.3f}, "
                  f"R2_a={med['r2_a']:.3f}, %Dev={med['pde']:.1f}")
        
        all_seed_results.append({'result': result, **med})
    
    avg = {
        'avg_r2_w': np.mean([s['r2_w'] for s in all_seed_results]),
        'avg_r2_a': np.mean([s['r2_a'] for s in all_seed_results]),
        'avg_pde':  np.mean([s['pde'] for s in all_seed_results]),
    }
    
    if verbose:
        print(f"\n  {model_type.upper()} averaged over {n_seeds} seeds:")
        print(f"    R2 weights:  {avg['avg_r2_w']:.3f}")
        print(f"    R2 activity: {avg['avg_r2_a']:.3f}")
        print(f"    % Deviance:  {avg['avg_pde']:.1f}")
    
    return {'seed_results': all_seed_results, 'model_type': model_type, **avg}


# ============ Plotting ============

def plot_figure3(taylor_result, mlp_result, save_path='figures/figure3.png'):
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    t_res = taylor_result['seed_results'][0]['result']
    m_res = mlp_result['seed_results'][0]['result']
    d = generate_ground_truth_behavior(seed=0)
    
    # Panel B: weight trajectory
    ax_b = fig.add_subplot(gs[0, 0])
    true_w = d['weights'][:, 0, 0].numpy()
    taylor_w = t_res['train_results'][0]['model_weights'][:, 0, 0]
    mlp_w = m_res['train_results'][0]['model_weights'][:, 0, 0]
    ax_b.plot(true_w, 'k-', lw=2, label='True Weight')
    ax_b.plot(taylor_w, 'b--', lw=1.5, label='Taylor')
    ax_b.plot(mlp_w, 'r:', lw=1.5, label='MLP')
    ax_b.set_xlabel('Trial'); ax_b.set_ylabel('Synaptic Weight')
    ax_b.set_title('B. Weight Dynamics (Single Synapse)'); ax_b.legend(fontsize=8)
    
    # Panel C: R2 distributions
    ax_c = fig.add_subplot(gs[0, 1])
    tr2_all = []
    mr2_all = []
    for s in taylor_result['seed_results']:
        tr2_all.extend([r['r2_weights'] for r in s['result']['train_results']])
    for s in mlp_result['seed_results']:
        mr2_all.extend([r['r2_weights'] for r in s['result']['train_results']])
    bp = ax_c.boxplot([tr2_all, mr2_all], tick_labels=['Taylor', 'MLP'],
                       patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax_c.set_ylabel('R² Score'); ax_c.set_title('C. R² Distributions (Weights)')
    ax_c.set_ylim(-0.1, 1.1)
    
    # Panel D: theta evolution
    ax_d = fig.add_subplot(gs[0, 2])
    ch = t_res['coeff_history']
    if ch:
        ch = np.array(ch)
        for a in range(3):
            for b in range(3):
                for g in range(3):
                    for dd in range(3):
                        if (a, b, g, dd) != (1, 0, 0, 1):
                            ax_d.plot(ch[:, a, b, g, dd], color='gray', alpha=0.15, lw=0.5)
        ax_d.plot(ch[:, 1, 0, 0, 1], color='red', lw=2.5,
                  label=r'$\theta_{1001}$ ($x_j \cdot r$)')
        ax_d.axhline(1, color='g', ls=':', alpha=0.5)
        ax_d.axhline(0, color='k', alpha=0.2)
    ax_d.set_xlabel('Epochs'); ax_d.set_ylabel(r'$\theta$ Value')
    ax_d.set_title('D. Coefficient Evolution'); ax_d.legend(fontsize=9)
    
    # Panel E: final coefficients
    ax_e = fig.add_subplot(gs[1, 0])
    final_coeffs = [s['result']['coeff_history'][-1]
                    for s in taylor_result['seed_results']
                    if s['result']['coeff_history']]
    if final_coeffs:
        fc_mean = np.mean(final_coeffs, axis=0)
        fc_std = np.std(final_coeffs, axis=0)
        terms = {
            'bias': (fc_mean[0,0,0,0], fc_std[0,0,0,0]),
            '$x_j$': (fc_mean[1,0,0,0], fc_std[1,0,0,0]),
            '$w_{ij}$': (fc_mean[0,0,1,0], fc_std[0,0,1,0]),
            '$r$': (fc_mean[0,0,0,1], fc_std[0,0,0,1]),
            '$x_j r$': (fc_mean[1,0,0,1], fc_std[1,0,0,1]),
        }
        names = list(terms.keys())
        vals = [terms[n][0] for n in names]
        errs = [terms[n][1] for n in names]
        colors = ['gray'] * 4 + ['red']
        ax_e.bar(range(len(names)), vals, yerr=errs, color=colors, capsize=3)
        ax_e.set_xticks(range(len(names)))
        ax_e.set_xticklabels(names, fontsize=9)
        ax_e.set_ylabel(r'$\theta$ Value')
        ax_e.set_title('E. Final Plasticity Coefficients')
        ax_e.axhline(0, color='k', lw=0.5)
    
    # Panel F: deviance explained
    ax_f = fig.add_subplot(gs[1, 1])
    tp_all = []
    mp_all = []
    for s in taylor_result['seed_results']:
        tp_all.extend([r['pde'] for r in s['result']['train_results']])
    for s in mlp_result['seed_results']:
        mp_all.extend([r['pde'] for r in s['result']['train_results']])
    bp2 = ax_f.boxplot([tp_all, mp_all], tick_labels=['Taylor', 'MLP'],
                        patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightsalmon')
    ax_f.set_ylabel('% Deviance Explained')
    ax_f.set_title('F. Behavioral Fit (% Deviance Explained)')
    
    # Summary
    ax_s = fig.add_subplot(gs[1, 2]); ax_s.axis('off')
    txt = (f"Averaged over {len(taylor_result['seed_results'])} seeds\n"
           f"{'='*35}\n\n"
           f"            R2_w   R2_a   %%Dev\n"
           f"Taylor:    {taylor_result['avg_r2_w']:.3f}  {taylor_result['avg_r2_a']:.3f}  {taylor_result['avg_pde']:.1f}\n"
           f"MLP:       {mlp_result['avg_r2_w']:.3f}  {mlp_result['avg_r2_a']:.3f}  {mlp_result['avg_pde']:.1f}\n\n"
           f"Paper (Table 1, rule x_j*r):\n"
           f"Taylor:    0.780  0.940  61.9\n"
           f"MLP:       0.850  0.960  64.8")
    ax_s.text(0.05, 0.5, txt, fontsize=10, family='monospace', va='center',
              transform=ax_s.transAxes)
    
    plt.suptitle("Figure 3: Recovery of Reward-Based Plasticity from Behavior",
                 fontsize=14, fontweight='bold', y=0.98)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


# ============ Main ============

if __name__ == '__main__':
    print("=" * 70)
    print("EXPERIMENT 2: Reward-Based Plasticity from Behavior (PyTorch)")
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)
    
    t0 = time.time()
    
    # Paper: num_epochs=250 in run.py, but they iterate over all 20 trajectories per epoch
    # 250 epochs * 20 traj = 5000 gradient steps
    # Our code does 1 traj per step, so 400 epochs * cycling = similar coverage
    
    print("\n[1/2] Training Taylor model (3 seeds x 400 epochs)...")
    taylor_result = train_behavior_model(
        model_type='taylor', n_epochs=400, lr=1e-3, n_seeds=3
    )
    
    print("\n[2/2] Training MLP model (3 seeds x 2000 epochs)...")
    mlp_result = train_behavior_model(
        model_type='mlp', n_epochs=2000, lr=1e-3, n_seeds=3
    )
    
    # Final comparison
    print(f"\n{'='*55}")
    print(f"RESULTS vs PAPER (Table 1, rule x_j*r)")
    print(f"{'='*55}")
    print(f"              R2_w  | R2_a  | %Dev")
    print(f"--------------+-------+-------+------")
    print(f"Taylor (ours) | {taylor_result['avg_r2_w']:.3f} | {taylor_result['avg_r2_a']:.3f} | {taylor_result['avg_pde']:.1f}")
    print(f"Taylor (paper)| 0.780 | 0.940 | 61.9")
    print(f"MLP (ours)    | {mlp_result['avg_r2_w']:.3f} | {mlp_result['avg_r2_a']:.3f} | {mlp_result['avg_pde']:.1f}")
    print(f"MLP (paper)   | 0.850 | 0.960 | 64.8")
    
    plot_figure3(taylor_result, mlp_result, save_path='figures/figure3.png')
    
    print(f"\nDone in {time.time()-t0:.1f}s")
