"""
Neural network simulation with synaptic plasticity (PyTorch).
FAST VERSION: Vectorized Taylor computation, no nested Python loops.

Matches paper's code (model.py) exactly:
  - Hidden layer: tanh activation
  - Output layer: fixed weights = 5.0/n_hidden, bias=0
  - Plasticity lr = 1.0 / input_dim
"""

import torch
import torch.nn.functional as F


def forward_pass(W, x):
    """y = sigmoid(W @ x) — used for Oja experiment only."""
    return torch.sigmoid(W @ x)


# ========== Oja experiment functions ==========

def simulate_oja_ground_truth(W0, inputs):
    T = inputs.shape[0]
    W = W0.clone()
    activities = []
    weights = []
    with torch.no_grad():
        for t in range(T):
            x = inputs[t]
            y = torch.sigmoid(W @ x)
            activities.append(y)
            weights.append(W.clone())
            dW = torch.outer(y, x) - (y**2).unsqueeze(1) * W
            W = W + dW
    return torch.stack(activities), torch.stack(weights)


def simulate_model_oja(W0, inputs, plasticity_rule):
    T = inputs.shape[0]
    W = W0.clone()
    activities = []
    weights = []
    for t in range(T):
        x = inputs[t]
        y = torch.sigmoid(W @ x)
        activities.append(y)
        weights.append(W)
        dW = plasticity_rule(x, y, W)
        W = W + dW
    return torch.stack(activities), torch.stack(weights)


# ========== FAST behavioral simulation ==========

def behavior_forward(W_plastic, x, n_hidden, last_layer_mult=5.0):
    """Forward pass: tanh hidden + fixed linear output."""
    hidden = torch.tanh(W_plastic @ x)
    logits = (last_layer_mult / n_hidden) * hidden.sum()
    prob = torch.sigmoid(logits)
    return hidden, logits, prob


def taylor_dw_fast(coeffs, x, y, w, r_val):
    """FAST Taylor plasticity: single vectorized operation.
    
    Replaces the 4-nested Python loop (81 iterations) with 
    precomputed einsum. ~10-50x faster.
    
    Args:
        coeffs: (3,3,3,3) Taylor coefficients
        x: (n_input,) presynaptic
        y: (n_hidden,) postsynaptic  
        w: (n_hidden, n_input) weights
        r_val: float, reward scalar
    """
    # Precompute powers as tensors
    x_p = torch.stack([torch.ones_like(x), x, x * x])           # (3, n_input)
    y_p = torch.stack([torch.ones_like(y), y, y * y])           # (3, n_hidden)
    w_p = torch.stack([torch.ones_like(w), w, w * w])           # (3, n_hidden, n_input)
    r_p = torch.tensor([1.0, r_val, r_val * r_val], 
                        device=w.device, dtype=w.dtype)          # (3,)
    
    # Single einsum: sum over all 81 terms at once
    # a=x_power, b=y_power, c=w_power, d=r_power, i=hidden, j=input
    dW = torch.einsum('abcd, aj, bi, cij, d -> ij',
                      coeffs, x_p, y_p, w_p, r_p)
    return dW


def mlp_dw_fast(fc1_weight, fc1_bias, fc2_weight, fc2_bias, x, y, w, r_val):
    """FAST MLP plasticity: batched forward pass.
    
    Args:
        fc1_weight, fc1_bias: hidden layer params
        fc2_weight, fc2_bias: output layer params
        x: (n_input,)
        y: (n_hidden,)
        w: (n_hidden, n_input)
        r_val: float
    """
    n_out, n_in = w.shape
    
    x_exp = x.unsqueeze(0).expand(n_out, -1)
    y_exp = y.unsqueeze(1).expand(-1, n_in)
    r_exp = torch.full_like(w, r_val)
    
    inp = torch.stack([x_exp, y_exp, w, r_exp], dim=-1)  # (n_out, n_in, 4)
    inp_flat = inp.reshape(-1, 4)
    
    h = F.leaky_relu(inp_flat @ fc1_weight.t() + fc1_bias)
    logits = h @ fc2_weight.t() + fc2_bias
    output = torch.tanh(logits)
    
    return output.squeeze(-1).reshape(n_out, n_in)


def simulate_behavior_model(W0, inputs, rewards, plasticity_rule,
                            moving_avg_window=10, use_reward_expectation=True):
    """Differentiable behavioral simulation with learned plasticity.
    
    Automatically detects Taylor vs MLP and uses fast path.
    """
    T = inputs.shape[0]
    n_hidden, n_input = W0.shape
    W = W0.clone()
    
    reward_avg = torch.tensor(0.0, device=W0.device, dtype=W0.dtype)
    alpha = 1.0 / moving_avg_window
    lr = 1.0 / n_input
    
    # Detect rule type for fast path
    is_taylor = hasattr(plasticity_rule, 'coeffs')
    is_mlp = hasattr(plasticity_rule, 'fc1')
    
    # Pre-extract MLP params if needed (avoid repeated attribute lookups)
    if is_mlp:
        fc1_w = plasticity_rule.fc1.weight
        fc1_b = plasticity_rule.fc1.bias
        fc2_w = plasticity_rule.fc2.weight
        fc2_b = plasticity_rule.fc2.bias
    elif is_taylor:
        coeffs = plasticity_rule.coeffs
    
    output_probs = []
    hidden_list = []
    weight_list = []
    
    for t in range(T):
        x = inputs[t]
        R = rewards[t]
        
        # Forward pass
        hidden = torch.tanh(W @ x)
        logits = (5.0 / n_hidden) * hidden.sum()
        prob = torch.sigmoid(logits)
        
        hidden_list.append(hidden)
        output_probs.append(prob)
        weight_list.append(W)
        
        if use_reward_expectation:
            r = R - reward_avg
        else:
            r = R
        
        reward_avg = ((1 - alpha) * reward_avg + alpha * R).detach()
        
        # FAST plasticity computation
        if isinstance(r, torch.Tensor):
            r_val = r.item()
        else:
            r_val = float(r)
        
        if is_taylor:
            dW = taylor_dw_fast(coeffs, x, hidden, W, r_val)
        elif is_mlp:
            dW = mlp_dw_fast(fc1_w, fc1_b, fc2_w, fc2_b, x, hidden, W, r_val)
        else:
            # Fallback: use module's forward (for Fly models etc)
            dW = plasticity_rule(x, hidden, W, r)
        
        W = W + lr * dW
        W = W.clamp(-10.0, 10.0)
    
    return torch.stack(output_probs), weight_list, torch.stack(hidden_list)
