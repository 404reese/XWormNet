import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _extract_lnn_internals(model, x):
    """Manually unroll LNN to extract gates and hidden states."""
    model.eval()
    with torch.no_grad():
        batch_size = x.size(0)
        u = model.input_layer(x)
        h = torch.zeros(batch_size, model.hidden_dim, device=x.device, dtype=x.dtype)
        
        a_vals = []
        b_vals = []
        h_vals = []
        
        for _ in range(model.num_steps):
            a_val = model.A_net(h)
            b_val = model.B_net(h)
            
            a_vals.append(a_val.cpu().numpy())
            b_vals.append(b_val.cpu().numpy())
            h_vals.append(h.cpu().numpy())
            
            dh = -a_val * h + b_val * u
            h = h + model.dt * dh
            
    return np.array(a_vals), np.array(b_vals), np.array(h_vals)

def plot_liquid_gates_over_time(model, X_sequence):
    os.makedirs('figures', exist_ok=True)
    a_vals, b_vals, _ = _extract_lnn_internals(model, X_sequence)
    
    a_mean = a_vals[:, 0, :].mean(axis=1)
    b_mean = b_vals[:, 0, :].mean(axis=1)
    
    plt.figure(figsize=(8, 4))
    plt.plot(a_mean, label='Gate A (Decay)', marker='o', color='crimson')
    plt.plot(b_mean, label='Gate B (Input)', marker='s', color='royalblue')
    plt.xlabel('Timestep (ODE Unrolling)')
    plt.ylabel('Mean Gate Value')
    plt.title('LNN Liquid Parameters Evolution Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/lnn_liquid_parameters.png', dpi=300)
    plt.close()

def plot_hidden_evolution_heatmap(model, X_sequence):
    os.makedirs('figures', exist_ok=True)
    _, _, h_vals = _extract_lnn_internals(model, X_sequence)
    
    h_data = h_vals[:, 0, :] 
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(h_data.T, cmap='magma', annot=False)
    plt.xlabel('Timestep (ODE Unrolling)')
    plt.ylabel('Hidden Unit Index')
    plt.title('LNN Hidden State Evolution Heatmap')
    plt.tight_layout()
    plt.savefig('figures/lnn_hidden_heatmap.png', dpi=300)
    plt.close()

def compare_lnn_vs_lstm(lnn_model, lstm_model, X_sequence):
    os.makedirs('figures', exist_ok=True)
    a_vals, _, _ = _extract_lnn_internals(lnn_model, X_sequence)
    
    plt.figure(figsize=(8, 4))
    
    for i in range(min(5, a_vals.shape[2])):
        plt.plot(a_vals[:, 0, i], label='LNN Time-varying Gates' if i==0 else None, color='dodgerblue', alpha=0.7, linewidth=2)
        
    if lstm_model is not None and hasattr(lstm_model, 'lstm'):
        lstm_w = lstm_model.lstm.weight_hh_l0.detach().cpu().numpy()
        lstm_mean_w = np.abs(lstm_w).mean()
        plt.axhline(y=lstm_mean_w, color='tomato', linestyle='--', linewidth=2, label='LSTM Fixed Weight (Mean)')
    else:
        plt.axhline(y=0.5, color='tomato', linestyle='--', linewidth=2, label='LSTM Fixed Weight (Representative)')
        
    plt.xlabel('Timestep (ODE Unrolling)')
    plt.ylabel('Parameter Value')
    plt.title('LNN Liquid Parameters vs LSTM Fixed Weights')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/lnn_vs_lstm_parameters.png', dpi=300)
    plt.close()
