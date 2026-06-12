import torch
import torch.nn as nn

class AutoregressiveClassifier(nn.Module):
    """
    Autoregressive (AR) model for sequence classification.
    Uses the last `ar_order` time steps of the sequence to predict the label.
    """
    def __init__(self, window_size, input_dim, ar_order=5):
        super(AutoregressiveClassifier, self).__init__()
        self.ar_order = min(ar_order, window_size)
        self.ar_layer = nn.Linear(self.ar_order * input_dim, 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        # Extract the last `ar_order` steps
        x_ar = x[:, -self.ar_order:, :]
        
        # Flatten for the linear AR layer
        x_flat = x_ar.reshape(x.size(0), -1)
        
        out = self.ar_layer(x_flat)
        out = self.relu(out)
        out = self.fc(out)
        return self.sigmoid(out)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
