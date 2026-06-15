import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    Vanilla RNN Classifier for binary traffic classification.

    Architecture:
    - Uses torch.nn.RNN (Elman RNN) as the recurrent backbone.
    - 2 stacked RNN layers with hidden_size=64, batch_first=True.
    - No dropout or gating mechanisms — this is the 'vanilla' baseline,
      intentionally simpler than LSTM/GRU to serve as a lower-bound
      reference in the 8-model comparison.
    - Final fully-connected layer maps the last hidden state → 1 output.
    - Sigmoid activation produces a probability in [0, 1] for binary
      classification (worm vs. benign traffic).

    Compared to LSTM (input gate, forget gate, output gate, cell state) and
    GRU (reset gate, update gate), vanilla RNN has no gating at all, making
    it faster but more susceptible to vanishing gradients over long sequences.
    It represents the theoretical minimum of recurrent complexity.
    """

    def __init__(self, input_dim: int, hidden_size: int = 64):
        """
        Initialise the RNNClassifier.

        Args:
            input_dim   (int): Number of input features per time-step.
            hidden_size (int): Number of hidden units in each RNN layer.
                               Defaults to 64.
        """
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            nonlinearity='tanh',   # standard Elman RNN activation
            dropout=0.0,           # no dropout — pure vanilla baseline
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Shape (batch, seq_len, input_dim)

        Returns:
            Tensor: Shape (batch, 1) — predicted probability of being malicious.
        """
        # rnn_out: (batch, seq_len, hidden_size)
        # hn     : (num_layers, batch, hidden_size)
        _, hn = self.rnn(x)
        # Take the last layer's hidden state
        last_hidden = hn[-1, :, :]          # (batch, hidden_size)
        out = self.fc(last_hidden)           # (batch, 1)
        return self.sigmoid(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience inference method (no gradient tracking).

        Args:
            x (Tensor): Shape (batch, seq_len, input_dim)

        Returns:
            Tensor: Shape (batch, 1) — predicted probabilities.
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
