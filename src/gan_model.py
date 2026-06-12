import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_size=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class AnomalyGAN(nn.Module):
    """
    GAN architecture for anomaly detection.
    The Generator learns to produce synthetic normal traffic.
    The Discriminator learns to distinguish real normal traffic from fake traffic.
    During inference, the Discriminator outputs the probability that traffic is normal.
    We return 1.0 - D(x) as the anomaly probability score.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_size=64):
        super(AnomalyGAN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, input_dim, hidden_size)
        self.discriminator = Discriminator(input_dim, hidden_size)

    def forward(self, x):
        d_out = self.discriminator(x)
        # Prob of anomaly = 1.0 - Prob of normal
        return 1.0 - d_out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
        return probs
