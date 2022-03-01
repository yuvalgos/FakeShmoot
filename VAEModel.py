import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.activation = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out + x


class VaeEncoder(nn.Module):
    def __init__(self, latent_size=128, num_residual_convs=3, device=None):
        super(VaeEncoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        feature_extractor_layers = [
            nn.PixelUnshuffle(2),  # -> 64x64x12
            nn.Conv2d(12, 16, kernel_size=3, padding='same'),  # -> 64x64x16
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),

            nn.PixelUnshuffle(2),  # -> 32x32x64
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),  # -> 32x32x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ]
        for _ in range(num_residual_convs):
            feature_extractor_layers.append(ResidualConvBlock(32))

        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        self.mu_fc = nn.Sequential(
            nn.Linear(32*32*32, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_size),)

        self.log_sigma_fc = nn.Sequential(
            nn.Linear(32*32*32, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_size),)

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
    def __init__(self, latent_size=128, num_residual_convs=3, device=None):
        super(VaeDecoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size
        self.feature_extractor = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 32*32*32),
            nn.BatchNorm1d(32*32*32),
            nn.LeakyReLU(),
        )

        decoder_layers = []
        for _ in range(num_residual_convs):
            decoder_layers.append(ResidualConvBlock(32))
        decoder_layers += [
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),  # -> 32x32x64
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.PixelShuffle(2),  # -> 64x64x16
            nn.Conv2d(16, 12, kernel_size=3, padding='same'),  # -> 64x64x12

            nn.PixelShuffle(2),  # -> 128x128x3
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        features = self.feature_extractor(z)
        features = features.view(features.size(0), 32, 32, 32)
        x_reconstructed = self.decoder(features)

        return x_reconstructed


class Vae(torch.nn.Module):
    def __init__(self, latent_size=128, num_residual_convs=3, device=None):
        super(Vae, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        self.encoder = VaeEncoder(latent_size, num_residual_convs=num_residual_convs, device=self.device)
        self.decoder = VaeDecoder(latent_size, num_residual_convs=num_residual_convs, device=self.device)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_sigma, z

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_size, device=self.device)
        fake_shmooots = self.decoder(z)

        return fake_shmooots
