import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM architecture for sequence-based network flow classification.
    Expects input sequences of shape (batch_size, window_size, input_dim).
    Uses a 2-layer LSTM with hidden size 64 and 20% dropout, followed by a linear
    layer and sigmoid activation for binary classification.
    """
    def __init__(self, input_dim, hidden_size=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output from the last time step
        # hn shape: (num_layers, batch_size, hidden_size)
        # We want the last layer's hidden state
        last_hidden = hn[-1, :, :]
        
        out = self.fc(last_hidden)
        return self.sigmoid(out)
        
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
