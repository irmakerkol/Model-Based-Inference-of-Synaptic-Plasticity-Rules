import torch
import torch.nn as nn

class CircuitModel(nn.Module):
    def __init__(self, n_input, n_output, plasticity_rule):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.plasticity_rule = plasticity_rule
        self.lr = 1.0

    def forward(self, X, W_init, observed_idx=None):
        B, T, _ = X.shape
        W = W_init.clone()
        m_traj = []

        for t in range(T):
            x_t = X[:, t, :]
            pre = torch.einsum('boi,bi->bo', W, x_t)
            y_t = torch.sigmoid(pre)

            m = y_t[:, observed_idx] if observed_idx is not None else y_t
            m_traj.append(m)

            
            y_sparse = torch.zeros_like(y_t)
            if observed_idx is not None:
                y_sparse[:, observed_idx] = y_t[:, observed_idx] # Add the colons!
                mask = torch.zeros_like(W)
                mask[:, observed_idx, :] = 1.0
                W = W * mask
            else:
                y_sparse = y_t
            
            dW = self.plasticity_rule(x_t, y_sparse, W, observed_idx=observed_idx)
        
            W = W + self.lr * dW

        return torch.stack(m_traj, dim=1)