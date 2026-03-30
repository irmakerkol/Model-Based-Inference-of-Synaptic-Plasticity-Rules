"""
Parameterized plasticity rules: Taylor series expansion and MLP.

PyTorch implementation of Equations (6) and (9) from the paper.
Matches paper's synapse.py exactly.

Taylor: volterra_plasticity_function with coefficients tensor
MLP: mlp_forward_pass with leaky_relu hidden + tanh output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaylorRule3Var(nn.Module):
    """Taylor series plasticity rule with 3 variables (x, y, w).
    Used for Oja's rule recovery (neural activity fitting).
    27 learnable coefficients.
    """
    def __init__(self, init_scale=1e-2):
        super().__init__()
        self.coeffs = nn.Parameter(torch.randn(3, 3, 3) * init_scale)
    
    def forward(self, x, y, w):
        x_powers = torch.stack([torch.ones_like(x), x, x**2])
        y_powers = torch.stack([torch.ones_like(y), y, y**2])
        w_powers = torch.stack([torch.ones_like(w), w, w**2])
        dW = torch.einsum('abc, aj, bi, cij -> ij', 
                          self.coeffs, x_powers, y_powers, w_powers)
        return dW


class TaylorRule4Var(nn.Module):
    """Taylor series plasticity rule with 4 variables (x, y, w, r).
    81 learnable coefficients.
    
    Paper's synapse.py init: scale=1e-5 for random init.
    """
    def __init__(self, init_scale=1e-5):
        super().__init__()
        # Paper: generate_gaussian(key, (3,3,3,3), scale=1e-5)
        self.coeffs = nn.Parameter(torch.randn(3, 3, 3, 3) * init_scale)
    
    def forward(self, x, y, w, r):
        device = w.device
        coeffs = self.coeffs.to(device)
        
        x_p = [torch.ones_like(x), x, x**2]
        y_p = [torch.ones_like(y), y, y**2]
        w_p = [torch.ones_like(w), w, w**2]
        
        if isinstance(r, torch.Tensor):
            r_val = r.item()
        else:
            r_val = float(r)
        r_p = [1.0, r_val, r_val**2]
        
        dW = torch.zeros_like(w)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        coeff = coeffs[a, b, c, d]
                        term = coeff * r_p[d]
                        term = term * x_p[a].unsqueeze(0) * y_p[b].unsqueeze(1) * w_p[c]
                        dW = dW + term
        return dW


class MLPRule(nn.Module):
    """MLP plasticity rule matching paper's synapse.py exactly.
    
    Paper's mlp_forward_pass:
        activation = leaky_relu(x @ w + b)   # hidden layers
        logits = activation @ final_w + final_b
        output = tanh(logits)                  # tanh at output
    
    Paper's init: generate_gaussian(key, shape, scale=0.01)
    """
    def __init__(self, input_dim=4, hidden_dim=10):
        super().__init__()
        # Paper uses manual params, we use nn.Linear for convenience
        # but match their init: scale=0.01 for all weights and biases
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Paper: generate_gaussian(key, shape, scale=0.01)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.normal_(self.fc1.bias, 0, 0.01)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.normal_(self.fc2.bias, 0, 0.01)
    
    def forward(self, x, y, w, r):
        n_out, n_in = w.shape
        
        x_exp = x.unsqueeze(0).expand(n_out, -1)
        y_exp = y.unsqueeze(1).expand(-1, n_in)
        
        if isinstance(r, torch.Tensor):
            r_val = r.item()
        else:
            r_val = float(r)
        r_exp = torch.full_like(w, r_val)
        
        inp = torch.stack([x_exp, y_exp, w, r_exp], dim=-1)  # (n_out, n_in, 4)
        inp_flat = inp.reshape(-1, 4)
        
        # Paper: leaky_relu hidden, tanh output
        h = F.leaky_relu(self.fc1(inp_flat))
        logits = self.fc2(h)
        output = torch.tanh(logits)  # Paper: output = jnp.tanh(logits)
        
        dW = output.squeeze(-1).reshape(n_out, n_in)
        return dW


class FlyPlasticityWithW(nn.Module):
    """Fly plasticity model WITH weight-dependent term.
    dw_ij = bias + w*w_ij + x*x_j + r*r + xr*x_j*r
    5 learnable parameters.
    """
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.w_coeff = nn.Parameter(torch.tensor(0.0))
        self.x_coeff = nn.Parameter(torch.tensor(0.0))
        self.r_coeff = nn.Parameter(torch.tensor(0.0))
        self.xr_coeff = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x, y, w, r):
        n_out, n_in = w.shape
        x_exp = x.unsqueeze(0).expand(n_out, -1)
        dW = (self.bias + 
              self.w_coeff * w + 
              self.x_coeff * x_exp + 
              self.r_coeff * r + 
              self.xr_coeff * x_exp * r)
        return dW


class FlyPlasticityWithoutW(nn.Module):
    """Fly plasticity model WITHOUT weight-dependent term.
    dw_ij = bias + x*x_j + r*r + xr*x_j*r
    4 learnable parameters.
    """
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.x_coeff = nn.Parameter(torch.tensor(0.0))
        self.r_coeff = nn.Parameter(torch.tensor(0.0))
        self.xr_coeff = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, x, y, w, r):
        n_out, n_in = w.shape
        x_exp = x.unsqueeze(0).expand(n_out, -1)
        dW = (self.bias + 
              self.x_coeff * x_exp + 
              self.r_coeff * r + 
              self.xr_coeff * x_exp * r)
        return dW


# --- Ground truth rules ---

def ojas_rule(x, y, w):
    """Oja's rule: dW = outer(y,x) - diag(y^2) @ W"""
    return torch.outer(y, x) - (y**2).unsqueeze(1) * w


def reward_covariance_rule(x, w, r, n_hidden):
    """dW = ones * x * r"""
    return x.unsqueeze(0).expand(n_hidden, -1) * r