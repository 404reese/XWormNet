import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, hn = self.gru(x)
        last_hidden = hn[-1, :, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
