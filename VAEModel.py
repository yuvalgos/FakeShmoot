import torch
from torch import nn


class VaeEncoder(torch.nn.Module):
    def __init__(self, latent_size=128, device=None):
        super(VaeEncoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> 64x64x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.mu_fc = nn.Sequential(
            nn.Linear(32*32*32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_size),)

        self.log_sigma_fc = nn.Sequential(
            nn.Linear(32*32*32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_size),)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        mu = self.mu_fc(features)
        log_sigma = self.log_sigma_fc(features)

        # sample from normal distribution and reparameterize:
        z = torch.rand_like(mu)
        z = z * torch.exp(0.5 * log_sigma) + mu

        return z, mu, log_sigma


class VaeDecoder(torch.nn.Module):
    def __init__(self, latent_size=128, device=None):
        super(VaeDecoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 32*32*32),
            nn.BatchNorm1d(32*32*32),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 64x64x16
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 128x128x3
            nn.Sigmoid(),
        )

    def forward(self, z):
        features = self.feature_extractor(z)
        features = features.view(features.size(0), 32, 32, 32)
        x_reconstructed = self.decoder(features)

        return x_reconstructed


class Vae(torch.nn.Module):
    def __init__(self, latent_size=128, device=None):
        super(Vae, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        self.encoder = VaeEncoder(latent_size, device=self.device)
        self.decoder = VaeDecoder(latent_size, device=self.device)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_sigma, z

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_size, device=self.device)
        fake_shmooots = self.decoder(z)

        return fake_shmooots
