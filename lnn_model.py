import torch
import torch.nn as nn

class LNN(nn.Module):
    r"""
    Liquid Neural Network (LNN) module.
    
    This model implements a continuous-time recurrent dynamical system approximated
    via a discrete-time forward Euler method.
    
    Mathematical Formulation:
        Continuous-time ODE:
            dx/dt = -A(x, t) * x + B(x, t) * u(t)
            
        Where:
            - x: system state (represented by the hidden state 'h')
            - u(t): input signal at time t (input features projected to hidden space)
            - A(x, t): state-dependent leakage/decay rate MLP (must be positive for stability)
            - B(x, t): state-dependent input coupling MLP
            
        Discrete-time Forward Euler approximation:
            h_{t+1} = h_t + dt * (-A(h_t) * h_t + B(h_t) * u)
            
        Where:
            - h: hidden state tensor of shape (batch_size, hidden_dim)
            - u: input representation of shape (batch_size, hidden_dim)
            - dt: time step size for numerical integration
            - '*' (or \odot): element-wise multiplication
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_steps: int = 6, dt: float = 0.1):
        """
        Initialize the LNN.
        
        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden state dynamics.
            num_steps (int): Number of Euler simulation time steps per forward pass.
            dt (float): Integration step size.
        """
        super(LNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dt = dt
        
        # 1. Input layer: projects raw input features to the hidden state space
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 2. A(h): Leakage network. Small MLP mapping state 'h' to decay rates.
        # We apply a Softplus activation to ensure the decay rate A(h) is strictly
        # positive, keeping the system stable and preventing exponential explosion.
        self.A_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Softplus()
        )
        
        # 3. B(h): Input coupling network. Small MLP mapping state 'h' to coupling strengths.
        self.B_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 4. Output layer: maps the final hidden state to a binary classification probability
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LNN.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Classification predictions of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Project raw input features x into the hidden state space dimension (u)
        u = self.input_layer(x)
        
        # Initialize hidden state h as zeros
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Simulate time steps using the Euler method loop
        for _ in range(self.num_steps):
            # Compute A(h) and B(h) based on the current hidden state h
            a_val = self.A_net(h)
            b_val = self.B_net(h)
            
            # Compute change in state: dh/dt = -A(h) * h + B(h) * u
            dh = -a_val * h + b_val * u
            
            # Apply Euler update step: h = h + dt * dh
            h = h + self.dt * dh
            
        # Map the final integrated state h to binary output probability
        output = self.output_layer(h)
        return output

if __name__ == "__main__":
    # Test execution script
    print("Testing Liquid Neural Network (LNN) implementation...")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    batch_size = 8
    input_dim = 10
    hidden_dim = 16
    num_steps = 6
    dt = 0.1
    
    # Initialize model
    model = LNN(input_dim=input_dim, hidden_dim=hidden_dim, num_steps=num_steps, dt=dt)
    print(f"Model initialized: Input Dim={input_dim}, Hidden Dim={hidden_dim}, Time Steps={num_steps}, dt={dt}")
    
    # Generate dummy input tensor
    dummy_input = torch.randn(batch_size, input_dim)
    print(f"Input tensor shape: {dummy_input.shape}")
    
    # Run forward pass
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}")
    
    # Verify shape
    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, f"Error: Output shape is {output.shape}, expected {expected_shape}"
    print("Output shape verification: SUCCESS")
    
    # Run backward pass to verify gradient propagation
    print("Testing backward pass gradient flow...")
    loss_fn = nn.BCELoss()
    dummy_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)
    loss = loss_fn(output, dummy_labels)
    loss.backward()
    
    # Check that all parameters have gradients computed
    all_grads_present = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  Warning: parameter {name} has no gradient!")
            all_grads_present = False
        else:
            print(f"  Parameter {name} gradient shape: {param.grad.shape}")
            
    if all_grads_present:
        print("Backward pass / gradient flow verification: SUCCESS")
    else:
        print("Backward pass / gradient flow verification: FAILED")
