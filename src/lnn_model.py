import torch
import torch.nn as nn

class LNN(nn.Module):
    r"""
    Liquid Neural Network (LNN) module.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_steps: int = 6, dt: float = 0.1):
        super(LNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dt = dt
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.A_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Softplus()
        )
        self.B_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        u = self.input_layer(x)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        for _ in range(self.num_steps):
            a_val = self.A_net(h)
            b_val = self.B_net(h)
            dh = -a_val * h + b_val * u
            h = h + self.dt * dh
        output = self.output_layer(h)
        return output

if __name__ == "__main__":
    import torch.nn as nn
    print("Testing LNN shim copy in src")
