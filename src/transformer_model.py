import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TrafficTransformer(nn.Module):
    """
    Transformer architecture for sequence-based network flow classification.
    Expects input sequences of shape (batch_size, window_size, input_dim).
    Uses a 2-layer Transformer Encoder with 8 heads, hidden size 64, and 10% dropout.
    """
    def __init__(self, input_dim, hidden_size=64, nhead=8, num_layers=2):
        super(TrafficTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(d_model=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        # Pooling: take the representation of the last time step
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
