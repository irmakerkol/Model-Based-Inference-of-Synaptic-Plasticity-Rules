# ─────────────────────────────────────────────────────────────────────────────
# DIFFUSION & FLOW MATCHING MODELS
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_ojas_data
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal 1D-CNN (Architecture tailored for Time Series)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalVectorFieldNet(nn.Module):
    """
    Predicts the vector field v(x_t, t) for Flow Matching.
    Uses 1D Convolutions to respect the temporal structure of Oja trajectories.
    Input shape: (Batch, N_neurons, Timesteps)
    """
    def __init__(self, n_neurons=50, hidden_dim=128):
        super().__init__()
        
        # Embedding to introduce time (t) to the network
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Convolutional filters sliding across the time series
        self.conv_in = nn.Conv1d(n_neurons, hidden_dim, kernel_size=3, padding=1)
        self.conv_hidden = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(hidden_dim, n_neurons, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        # x shape: (B, N_neurons, T)
        # t shape: (B, 1)
        
        # Calculate time embedding and broadcast along the time axis
        t_emb = self.t_embed(t).unsqueeze(-1)  # (B, hidden_dim, 1)
        
        # 1D Convolutional Network (Preserves the temporal flow)
        h = self.act(self.conv_in(x))
        h = h + t_emb  # Inject time information
        h = self.act(self.conv_hidden(h))
        out = self.conv_out(h)
        
        return out

# ─────────────────────────────────────────────────────────────────────────────
# 2. Continuous-Time Flow Matching
# ─────────────────────────────────────────────────────────────────────────────
class FlowMatcher:
    """
    Implements Optimal Transport Flow Matching.
    """
    def __init__(self, model):
        self.model = model

    def compute_loss(self, x_0):
        # x_0 (Clean Data) shape: (B, N_neurons, T)
        B = x_0.shape[0]
        
        # 1. Continuous time t ~ U(0, 1) (As in the professor's notes)
        t = torch.rand(B, 1, 1, device=x_0.device)
        
        # 2. Pure Gaussian Noise
        epsilon = torch.randn_like(x_0)
        
        # 3. Interpolation (PDF 2 Page 8: x_t = (1-t)x_0 + t * epsilon)
        x_t = (1 - t) * x_0 + t * epsilon
        
        # 4. Target Vector Velocity (PDF 2 Page 8: u = epsilon - x_0)
        target_v = epsilon - x_0
        
        # 5. Model's Velocity Prediction (Reshaping t vector to (B, 1))
        pred_v = self.model(x_t, t.view(B, 1))
        
        # 6. Loss (Squared Loss)
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def generate(self, shape, steps=100, device='cpu'):
        """
        Probability Flow ODE:
        Solves the ODE backwards using the Euler method.
        """
        self.model.eval()
        
        # 1. Start from pure noise (t=1)
        x = torch.randn(shape, device=device)
        
        # Step backwards from t=1 to t=0
        t_steps = torch.linspace(1.0, 0.0, steps, device=device)
        dt = 1.0 / steps
        
        for t_val in t_steps:
            t_batch = torch.full((shape[0], 1), t_val.item(), device=device)
            
            # Request the average velocity from the network
            v_pred = self.model(x, t_batch)
            
            # Euler ODE Step: dx = v(x, t) * dt
            x = x - v_pred * dt  # Minus (-) because we are moving backwards
            
        return x

# ─────────────────────────────────────────────────────────────────────────────
# 3. Training and Experiment Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_flow_matching_experiment():
    print("\nTraining Continuous-Time Flow Matching Model...")
    
    # 1. Generate Ground-Truth Oja Data (Uses the function already in notebook)
    X, O, W_gt, obs_idx = generate_ojas_data(
        n_input=100, n_output=50, T=50,
        n_trajectories=50, noise_std=0.0
    )
    
    # Move to default device if using GPU in Kaggle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    O = O.to(device)
    
    # O shape: (50, 50, 50) -> (Batch, Timesteps, N_neurons)
    # PyTorch Conv1d expects (Batch, Channels, Length), so we transpose.
    O_train = O.transpose(1, 2) # New shape: (50, 50 neurons, 50 timesteps)
    
    model = TemporalVectorFieldNet(n_neurons=50).to(device)
    fm = FlowMatcher(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    n_epochs = 1000
    batch_size = 16
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(O_train))
        epoch_loss = 0.0
        
        for i in range(0, len(O_train), batch_size):
            batch = O_train[perm[i:i+batch_size]]
            optimizer.zero_grad()
            
            loss = fm.compute_loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}/{n_epochs} | ODE Flow Loss: {epoch_loss:.4f}")

    print("\nSampling via Probability Flow ODE ...")
    # Generate 20 new time series
    synthetic_data = fm.generate(shape=(20, 50, 50), steps=100, device=device)
    
    # Clamp output to [0, 1] range and revert to original shape (Batch, T, Neurons)
    synthetic_data = synthetic_data.clamp(0, 1).transpose(1, 2)
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Flow Matching (ODE) Generated Neural Trajectories", fontsize=12)
    
    # Bringing back to CPU for matplotlib
    O_cpu = O.cpu()
    synth_cpu = synthetic_data.cpu()
    
    for i in range(3):
        # Real Data
        axes[0, i].plot(O_cpu[i, :, :5].numpy(), alpha=0.7)
        axes[0, i].set_title(f'Real trajectory {i}')
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_ylabel('Neural activity')
        axes[0, i].grid(True, alpha=0.3)
        
        # Synthetic Data
        axes[1, i].plot(synth_cpu[i, :, :5].numpy(), alpha=0.7)
        axes[1, i].set_title(f'Flow Matching {i}')
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_ylabel('Neural activity')
        axes[1, i].set_xlabel('Timestep t')
        axes[1, i].grid(True, alpha=0.3)
        
    # save images and files:
    image_path = 'week2_flow_matching_trajectories.png'
    data_path = 'synthetic_neural_responses.pt'

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.show()
    print(f"Saved image to: {image_path}")

    # Export the synthetic data
    torch.save(synthetic_data, data_path)
    print(f"Saved synthetic data to: {data_path}")
    
    return synthetic_data

if __name__ == "__main__":
    # to test Flow Matching:
    synthetic_data = run_flow_matching_experiment()
    print("Synthetic data shape:", synthetic_data.shape)