import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
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
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1, :, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
