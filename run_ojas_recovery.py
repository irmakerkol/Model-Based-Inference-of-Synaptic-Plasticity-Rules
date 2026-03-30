# week2_ojas_recovery.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data_generation import generate_ojas_data
from src.plasticity_rules import  TaylorRule3Var, MLPPlasticityRule
from circuit_model import CircuitModel


# ─────────────────────────────────────────────────────────────────────────────
# Full Oja's Recovery Experiment
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(f"Global default device set to: {device}")

def mse_loss(m_traj, o_traj):

    return ((o_traj - m_traj) ** 2).mean()

def run_ojas_recovery(
    n_input= 100, n_output=1000, T=50,
    n_trajectories=50, n_epochs=250,
    noise_std=0.0, sparsity=1.0,
    lr_optimizer=1e-3, grad_clip=0.2,
    seed=42, verbose=True
):
    torch.manual_seed(seed)
 
    # ── 1.Generate  data ───────────────────────────────────────
    X_train, O_train, W_gt_train, obs_idx = generate_ojas_data(
        n_input, n_output, T, n_trajectories, noise_std, sparsity, seed
    )
    X_train     = X_train.to(device)
    O_train     = O_train.to(device)
    W_gt_train  = W_gt_train.to(device)
    obs_idx     = obs_idx.to(device)
    W_inits_train = W_gt_train[:, 0]  
 
    # ── 2. Build model ────────────────────────────────────────────────────────
    rule    = TaylorRule3Var().to(device)   # Start with the Taylor expansion model
    #rule    = MLPPlasticityRule().to(device) 
    circuit = CircuitModel(n_input, n_output, rule).to(device)
    optimizer = optim.Adam(rule.parameters(), lr=lr_optimizer)
 
    mask_other = np.ones((3, 3, 3), dtype=bool)
    mask_other[1, 1, 0] = False   # Exclude Hebbian  θ_110
    mask_other[0, 2, 1] = False   # Exclude Decay    θ_021
 
    history = {
        'weight_error_over_time': [],
        'theta_110':    [],
        'theta_021':    [],
        'other_thetas': [],
    }
 

    # ── 3. Training loop ──────────────────────────────────────────────────────
    for epoch in range(n_epochs):
 
        # Accumulate loss on GPU 
        epoch_loss_gpu = torch.zeros(1, device=device)
 
        perm = torch.randperm(len(X_train), device=device)
 
        optimizer.zero_grad(set_to_none=True)
 
        for i in range(len(X_train)):
            idx = perm[i].item()
 
            X_batch      = X_train[idx:idx+1]
            O_batch      = O_train[idx:idx+1]
            W_init_batch = W_inits_train[idx:idx+1]
            
            
            m_pred = circuit.forward(
                X_batch, W_init=W_init_batch, observed_idx=obs_idx
            )
            loss = mse_loss(m_pred, O_batch)
            loss.backward()
 
            torch.nn.utils.clip_grad_norm_(rule.parameters(), grad_clip)
            optimizer.step()
 
            
            epoch_loss_gpu += loss.detach()
 
            # zero_grad for next trajectory
            optimizer.zero_grad()
 
        # Single GPU→CPU sync per epoch 
        epoch_loss = (epoch_loss_gpu / len(X_train)).item()
        
        # ── Coefficient tracking ─────
        t_vals = rule.coeffs.detach().cpu().numpy()
        history['theta_110'].append(t_vals[1, 1, 0])
        history['theta_021'].append(t_vals[0, 2, 1])
        history['other_thetas'].append(t_vals[mask_other].tolist())
 
        # ── Weight-error heatmap (every 50 epochs — uses full W_traj) ─────
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                _, W_pred = circuit_forward_with_weights(
                    circuit, X_train, W_inits_train, observed_idx=obs_idx
                )
                min_T    = min(W_pred.shape[1], W_gt_train.shape[1])
                err_per_t = (
                    (W_pred[:, :min_T] - W_gt_train[:, :min_T]) ** 2
                ).mean(dim=(0, 2, 3))
                history['weight_error_over_time'].append(
                    err_per_t.cpu().numpy().tolist()
                )
        else:
            history['weight_error_over_time'].append(
                history['weight_error_over_time'][-1]
            )
 
        if verbose and epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | Activity Loss: {epoch_loss:.6f} "
                f"| Final W-MSE: {history['weight_error_over_time'][-1][-1]:.6f}"
            )
        
    # ── 4. Evaluate R² on fresh test trajectories ─────────────────────────────
    rule.eval()
    with torch.no_grad():
        test_seed = seed + 999
 

        X_test, _, W_gt_test, obs_idx_test = generate_ojas_data(
            n_input=n_input, n_output=n_output, T=T,
            n_trajectories=n_trajectories,
            noise_std=noise_std,   
            sparsity=sparsity,    
            seed=test_seed
        )
        X_test       = X_test.to(device)
        W_gt_test    = W_gt_test.to(device)
        obs_idx_test = obs_idx_test.to(device)
        W_inits_test = W_gt_test[:, 0]   # (B, n_output, n_input)
 
        
        _, W_pred_test = circuit_forward_with_weights(
            circuit, X_test, W_inits_test, observed_idx=obs_idx_test
        )
 
        mean_r2 = compute_r2(W_pred_test, W_gt_test, W_inits_test, observed_idx=None)
        del W_pred_test, X_test, W_gt_test, X_train, O_train # Delete large objects
        torch.cuda.empty_cache()
 
    if verbose:
 
        print(f"\nMean R² on fresh test weight trajectories (ΔW): {mean_r2:.4f}")
 
    return history, mean_r2, rule

def circuit_forward_with_weights(circuit, X, W_init, observed_idx=None):
    B, T, _ = X.shape
    W = W_init.clone()
    m_traj = []
    W_traj = [W.clone()]

    for t in range(T):
        x_t = X[:, t, :]
        
        y_t = torch.sigmoid(torch.einsum('boi,bi->bo', W, x_t))
        
        y_sparse = torch.zeros_like(y_t)
        if observed_idx is not None:
            y_sparse[:, observed_idx] = y_t[:, observed_idx] 
            mask = torch.zeros_like(W)
            mask[:, observed_idx, :] = 1.0
            W = W * mask
        else:
            y_sparse = y_t
        
        m = y_t[:, observed_idx] if observed_idx is not None else y_t
        m_traj.append(m)

        
        dW = circuit.plasticity_rule(x_t, y_sparse, W, observed_idx=observed_idx)
        W = W + circuit.lr * dW
        W_traj.append(W.clone())

    return torch.stack(m_traj, dim=1), torch.stack(W_traj, dim=1)




    
def compute_r2(W_pred, W_true, W_init, observed_idx=None):
    W_pred = W_pred.cpu()
    W_true = W_true.cpu()
    W_init = W_init.cpu()

    if observed_idx is not None:
        observed_idx = observed_idx.cpu()
        W_pred = W_pred[:, :, observed_idx, :]
        W_true = W_true[:, :, observed_idx, :]
        W_init = W_init[:, observed_idx, :]   

    W_init_expanded = W_init.unsqueeze(1)

    ss_res = ((W_true - W_pred) ** 2).sum()

    delta_true = W_true - W_init_expanded
    ss_tot = ((delta_true - delta_true.mean()) ** 2).sum()

    return (1 - ss_res / (ss_tot + 1e-10)).item()




# ==============================================================================
# EXPERIMENT RUNNERS
# ==============================================================================
def run_dynamics_experiment():
    print("Running Training Dynamics (Panels B, C, G)...")
    # Run one long, clean training session for the line graphs and heatmap
    history, _, _ = run_ojas_recovery(
        noise_std=1.0, lr_optimizer=1e-3, sparsity=1.0, n_epochs=400, verbose=True
    )

    history['other_thetas'] = np.array(history['other_thetas']).T.tolist()
    return history


def run_robustness_grid():
    print("Running Grid Search for Robustness (Panels D, E, F)...")
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sparsity_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    seeds = [10,100]  

    r2_matrix = np.zeros((len(noise_levels), len(sparsity_levels)))
    r2_distributions = {}

    total = len(noise_levels) * len(sparsity_levels) * len(seeds)
    done = 0

    for i, noise in enumerate(noise_levels):
        for j, sparsity in enumerate(sparsity_levels):
            scores = []
            for seed in seeds:
                print(f"[{done+1}/{total}] Running Noise: {noise}, Sparsity: {sparsity}, Seed: {seed}")
                
                
                _, r2, _ = run_ojas_recovery(
                    lr_optimizer=1e-3,
                    noise_std=noise,
                    sparsity=sparsity,
                    n_epochs=250, 
                    n_trajectories=50,
                    seed=seed,
                    verbose=True
                )
                scores.append(r2)
                done += 1
                torch.cuda.empty_cache() 

            r2_matrix[i, j] = np.mean(scores)
            r2_distributions[(noise, sparsity)] = scores
            print(f"  -> R² = {np.mean(scores):.3f}")

    return noise_levels, sparsity_levels, r2_matrix, r2_distributions


def plot_theta_trajectories(history):
    """
    Standalone plot to visualize only the convergence of the Taylor coefficients.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    coral = '#FF7F50'

    epochs = np.arange(len(history['theta_110']))
    
    # Ground truth reference lines
    ax.axhline(1.0, color='lightgreen', linestyle='--', zorder=1)
    ax.axhline(0.0, color='lightgreen', linestyle='--', zorder=1)
    ax.axhline(-1.0, color='lightgreen', linestyle='--', zorder=1)

    # Plot all the non-Oja theta coefficients (the ones that should go to 0)
    for ot in history['other_thetas']:
        ax.plot(epochs, ot, color='dimgray', linewidth=1.5, zorder=2)
        
    # Highlight the target Oja's rule coefficients
    ax.plot(epochs, history['theta_110'], color=coral, linewidth=2, zorder=3)
    ax.plot(epochs, history['theta_021'], color=coral, linewidth=2, zorder=3)

    # Add text labels tracking the target curves
    ax.text(epochs[-1] * 0.8, 0.8, r'$\theta_{110}$', color=coral, fontsize=14, fontweight='bold')
    ax.text(epochs[-1] * 0.8, -0.8, r'$\theta_{021}$', color=coral, fontsize=14, fontweight='bold')
    
    # Formatting
    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel(r"$\theta_{\alpha,\beta,\gamma}$ Value", fontsize=12)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Taylor Coefficient Convergence", fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/theta_convergence.png", dpi=200)
    plt.show()

def plot_all_figures(history, noise_levels, sparsity_levels, r2_matrix, r2_distributions):
    fig = plt.figure(figsize=(16, 9))
    coral = '#FF7F50'

    ax_B = fig.add_subplot(231)
    err_array = np.array(history['weight_error_over_time'])
    im = ax_B.imshow(err_array, aspect='auto', cmap='viridis', origin='upper',
                     extent=[0, err_array.shape[1], err_array.shape[0], 0])
    ax_B.set_xlabel("Time"); ax_B.set_ylabel("Training Epochs"); ax_B.set_title("B", loc='left', fontweight='bold')
    fig.colorbar(im, ax=ax_B, pad=0.02).set_label("Weight MSE")

    # --- PANEL C/G: Theta Trajectories ---
    ax_C = fig.add_subplot(232)
    epochs = np.arange(len(history['theta_110']))
    ax_C.axhline(1.0, color='lightgreen', linestyle='--', zorder=1)
    ax_C.axhline(0.0, color='lightgreen', linestyle='--', zorder=1)
    ax_C.axhline(-1.0, color='lightgreen', linestyle='--', zorder=1)

    for ot in history['other_thetas']:
        ax_C.plot(epochs, ot, color='dimgray', linewidth=1.5, zorder=2)
    ax_C.plot(epochs, history['theta_110'], color=coral, linewidth=2, zorder=3)
    ax_C.plot(epochs, history['theta_021'], color=coral, linewidth=2, zorder=3)

    ax_C.text(epochs[-1]*0.8, 0.8, r'$\theta_{110}$', color=coral, fontsize=12)
    ax_C.text(epochs[-1]*0.8, -0.8, r'$\theta_{021}$', color=coral, fontsize=12)
    ax_C.set_xlabel("Training Epochs"); ax_C.set_ylabel(r"$\theta_{\alpha,\beta,\gamma}$ Value")
    ax_C.set_ylim(-1.2, 1.2); ax_C.set_title("C / G", loc='left', fontweight='bold')

    # --- PANEL D: R2 Heatmap ---
    ax_D = fig.add_subplot(234)
    im2 = ax_D.imshow(r2_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax_D.set_xticks(np.arange(len(sparsity_levels))); ax_D.set_xticklabels(sparsity_levels)
    ax_D.set_yticks(np.arange(len(noise_levels))); ax_D.set_yticklabels(noise_levels)
    rect = patches.Rectangle((-0.5, -0.5), len(sparsity_levels), 1, linewidth=2, edgecolor='deeppink', facecolor='none')
    ax_D.add_patch(rect)
    ax_D.set_xlabel("Sparseness"); ax_D.set_ylabel("Noise Scale"); ax_D.set_title("D", loc='left', fontweight='bold')
    fig.colorbar(im2, ax=ax_D).set_label(r"$\mathcal{R}^2$ score")

    # --- Helper for Boxplots ---
    def custom_boxplot(ax, data, positions, edge_color):
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.5, showfliers=True)
        for box in bp['boxes']: box.set(facecolor='#1E7145', color='black')
        for median in bp['medians']: median.set(color='black', linewidth=1.5)
        for spine in ax.spines.values(): spine.set_edgecolor(edge_color); spine.set_linewidth(2)

    
    baseline_sparsity = sparsity_levels[0]  
    baseline_noise = noise_levels[0] 

    # --- PANEL E: Boxplot by Noise (Sparsity = Baseline) ---
    ax_E = fig.add_subplot(235)
    data_e = [r2_distributions[(n, baseline_sparsity)] for n in noise_levels]
    custom_boxplot(ax_E, data_e, positions=np.arange(len(noise_levels)), edge_color='deepskyblue')
    ax_E.set_xticks(np.arange(len(noise_levels))); ax_E.set_xticklabels(noise_levels)
    ax_E.set_xlabel("Noise Scale"); ax_E.set_ylabel(r"$\mathcal{R}^2$ score")
    ax_E.set_ylim(-0.05, 1.05); ax_E.set_title("E", loc='left', fontweight='bold')
    # --- PANEL F: Boxplot by Sparseness (Noise = Baseline) ---
    ax_F = fig.add_subplot(236)
    data_f = [r2_distributions[(baseline_noise, s)] for s in sparsity_levels]
    custom_boxplot(ax_F, data_f, positions=np.arange(len(sparsity_levels)), edge_color='deeppink')
    ax_F.set_xticks(np.arange(len(sparsity_levels))); ax_F.set_xticklabels(sparsity_levels)
    ax_F.set_xlabel("Sparseness"); ax_F.set_ylabel(r"$\mathcal{R}^2$ score")
    ax_F.set_ylim(-0.05, 1.05); ax_F.set_title("F", loc='left', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/figure1.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 1. Run the deep dive for Panels B & C
    hist = run_dynamics_experiment()
    plot_theta_trajectories(hist)
    
    # 2. Run the massive sweep for Panels D, E & F
    n_levs, s_levs, r2_mat, r2_dists = run_robustness_grid()
    
    # 3. Graph everything in one beautiful Matplotlib window
    plot_all_figures(hist, n_levs, s_levs, r2_mat, r2_dists)