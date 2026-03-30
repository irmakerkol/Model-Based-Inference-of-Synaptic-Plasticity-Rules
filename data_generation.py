
import math
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return torch.sigmoid(x)


def visualize_trajectories(X, O, W, traj_idx=0):
    """Plot input stimulus, neural activity, and weight dynamics for one trajectory."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    T = X.shape[1]
    t = np.arange(T)
    
    # Panel 1: Input stimulus (first 5 neurons)
    ax = axes[0]
    for i in range(5):
        ax.plot(t, X[traj_idx, :, i].numpy(), alpha=0.7, label=f'x_{i}')
    ax.set_ylabel('Input activity x(t)')
    ax.set_title(f'Sample trajectory #{traj_idx} — Oja\'s rule synthetic data')
    ax.legend(fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Neural output (first 5 observed neurons)
    ax = axes[1]
    for i in range(5):
        ax.plot(t, O[traj_idx, :, i].numpy(), alpha=0.7, label=f'y_{i}')
    ax.set_ylabel('Neural output y(t)')
    ax.set_ylim(0, 1)  # sigmoid output should be in [0,1]
    ax.legend(fontsize=7, ncol=5)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Weight trajectory (mean absolute weight over time)
    ax = axes[2]
    w_mean = W[traj_idx].abs().mean(dim=(1, 2)).numpy()  # (T+1,)
    ax.plot(np.arange(T + 1), w_mean, color='darkblue')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Mean |w_ij|')
    ax.set_title('Weight magnitude over time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_trajectories.png', dpi=150)
    plt.show()
    print("Saved: week1_trajectories.png")

def ojas_rule(x, y, W):
    """
    Oja's plasticity rule: Δw_ij = x_j * y_i - y_i^2 * w_ij
    
    Args:
        x: presynaptic activity, shape (n_input,)
        y: postsynaptic activity, shape (n_output,)
        W: current weight matrix, shape (n_output, n_input)
    
    Returns:
        dW: weight update, shape (n_output, n_input)
    """
    # Outer product: y_i * x_j gives shape (n_output, n_input)
    hebbian_term = torch.outer(y, x)
    
    # Decay term: y_i^2 * w_ij 
    decay_term = (y ** 2).unsqueeze(1) * W  
    
    return hebbian_term - decay_term

def generate_ojas_data(
    n_input=100,
    n_output=1000,
    T=50,               # timesteps per trajectory
    n_trajectories=50,  # number of independent trajectories
    noise_std=0.0,      # additive Gaussian noise on observations
    sparsity=1.0,       # fraction of output neurons observed         
    seed=42
):
    """
    Generate synthetic neural activity trajectories using Oja's rule.
    
    Returns:
        X_all: input stimuli, shape (n_trajectories, T, n_input)
        O_all: observed neural outputs, shape (n_trajectories, T, n_observed)
        W_all: weight trajectories, shape (n_trajectories, T+1, n_output, n_input)
        observed_idx: which neuron indices are observed
    """
    torch.manual_seed(seed)
    # Which output neurons are observed (for sparsity experiments)
    n_observed = int(n_output * sparsity)
    observed_idx = torch.randperm(n_output)[:n_observed]
    

   
    
    X_all, O_all, W_all = [], [], []
    
    for traj in range(n_trajectories):
        # KAIMING initialization for weights
        std_w = (2.0 / n_input) ** 0.5
        W = torch.randn(n_output, n_input) * std_w
        
        X_traj, O_traj, W_traj = [], [], [W.clone()]
      
        
        for t in range(T):
            # Input: sampled from Gaussian with mean 0, var 0.1 (as in paper Appendix A.3)
            x = torch.randn(n_input) *  math.sqrt(0.1)
            
            # Forward pass
            y = sigmoid(W @ x)  # shape (n_output,)
            
            
            # Observe subset of neurons + add noise
            o = y[observed_idx].clone()
            if noise_std > 0:
                o = o + torch.randn_like(o) * noise_std
                o = torch.clamp(o, 0.0, 1.0)
            
            y_sparse = torch.zeros_like(y)
            y_sparse[observed_idx] = o
            
            # Weight update using Oja's rule
            dW = ojas_rule(x, y_sparse, W)
            W = W + dW 
            
            
            X_traj.append(x)
            O_traj.append(o)
            W_traj.append(W.clone())
        
        X_all.append(torch.stack(X_traj))    # (T, n_input)
        O_all.append(torch.stack(O_traj))    # (T, n_observed)
        W_all.append(torch.stack(W_traj))    # (T+1, n_output, n_input)
    
    return (
        torch.stack(X_all),   # (n_traj, T, n_input)
        torch.stack(O_all),   # (n_traj, T, n_observed)
        torch.stack(W_all),   # (n_traj, T+1, n_output, n_input)
        observed_idx
    )


# ── Run it and verify shapes ──────────────────────────────────────────────────
if __name__ == "__main__":
    X, O, W, obs_idx = generate_ojas_data()
    
    print("=== Data shapes ===")
    print(f"X (stimuli):      {X.shape}")   # expect (50, 50, 100)
    print(f"O (observations): {O.shape}")   # expect (50, 50, 50)
    print(f"W (weights):      {W.shape}")   # expect (50, 51, 50, 100)
    print(f"Observed neurons: {obs_idx.shape}")
    
    # Basic sanity checks
    assert not torch.isnan(X).any(), "NaNs in X!"
    assert not torch.isnan(O).any(), "NaNs in O!"
    assert not torch.isnan(W).any(), "NaNs in W!"
    print("✓ No NaNs detected")
    
    w_change = (W[:, -1] - W[:, 0]).abs().mean()
    print(f"✓ Mean absolute weight change over trajectory: {w_change:.4f}")
    assert w_change > 1e-6, "Weights did not change — plasticity rule may be broken!"
    
    print("\n=== Visualizing sample trajectories ===")
    visualize_trajectories(X, O, W)
