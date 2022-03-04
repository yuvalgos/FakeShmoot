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
    def __init__(self, latent_size=128, fc_layers=(512, 512, 512), num_residual_convs=3, device=None):
        super(VaeEncoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        feature_extractor_layers = [
            nn.PixelUnshuffle(2),  # -> 64x64x12
            nn.Conv2d(12, 12, kernel_size=3, padding='same'),  # -> 64x64x12
            nn.LeakyReLU(),
            nn.BatchNorm2d(12),

            nn.PixelUnshuffle(2),  # -> 48x32x32
            nn.Conv2d(48, 48, kernel_size=3, padding='same'),  # -> 48x32x32
            nn.LeakyReLU(),
            nn.BatchNorm2d(48),
        ]
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        residual_conv_layers = []
        for _ in range(num_residual_convs):
            residual_conv_layers.append(ResidualConvBlock(48))
        self.residual_conv = nn.Sequential(*residual_conv_layers)

        self.conv_last1x1 = nn.Conv2d(48, 24, kernel_size=1, padding='same')

        mu_fc_layers = [nn.Linear(24 * 32 * 32, fc_layers[0])]
        for i in range(1, len(fc_layers)):
            mu_fc_layers.append(nn.Linear(fc_layers[i - 1], fc_layers[i]))
            mu_fc_layers.append(nn.LeakyReLU())
            mu_fc_layers.append(nn.BatchNorm1d(fc_layers[i]))
        mu_fc_layers.append(nn.Linear(fc_layers[-1], latent_size))
        self.mu_fc = nn.Sequential(*mu_fc_layers)

        log_sigma_fc_layers = [nn.Linear(48 * 32 * 32, fc_layers[0])]
        for i in range(1, len(fc_layers)):
            log_sigma_fc_layers.append(nn.Linear(fc_layers[i - 1], fc_layers[i]))
            log_sigma_fc_layers.append(nn.LeakyReLU())
            log_sigma_fc_layers.append(nn.BatchNorm1d(fc_layers[i]))
        log_sigma_fc_layers.append(nn.Linear(fc_layers[-1], latent_size))
        self.log_sigma_fc = nn.Sequential(*log_sigma_fc_layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features + self.residual_conv(features)
        features = self.conv_last1x1(features)
        features = features.view(features.size(0), -1)

        mu = self.mu_fc(features)
        log_sigma = self.log_sigma_fc(features)

        # sample from normal distribution and reparameterize:
        z = torch.rand_like(mu)
        z = z * torch.exp(0.5 * log_sigma) + mu

        return z, mu, log_sigma


class VaeDecoder(torch.nn.Module):
    def __init__(self, latent_size=128, fc_layers=(512, 512), num_residual_convs=3, device=None):
        super(VaeDecoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        feature_extractor_layers = [nn.Linear(latent_size, fc_layers[0])]
        for i in range(1, len(fc_layers)):
            feature_extractor_layers.append(nn.Linear(fc_layers[i - 1], fc_layers[i]))
            feature_extractor_layers.append(nn.LeakyReLU())
            feature_extractor_layers.append(nn.BatchNorm1d(fc_layers[i]))
        feature_extractor_layers.append(nn.Linear(fc_layers[-1], 24 * 32 * 32))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        self.first_1x1conv = nn.Conv2d(24, 48, kernel_size=1, padding='same')

        residual_conv_layers = []
        for _ in range(num_residual_convs):
            residual_conv_layers.append(ResidualConvBlock(48))
        self.residual_conv = nn.Sequential(*residual_conv_layers)

        decoder_layers = [
            nn.PixelShuffle(2),  # -> 12x64x64
            nn.Conv2d(12, 12, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(12),

            nn.PixelShuffle(2),  # -> 3x128x128
            nn.Conv2d(3, 3, kernel_size=3, padding='same'),
            nn.Sigmoid(),]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        features = self.feature_extractor(z)
        features = features.view(features.size(0), 24, 32, 32)
        x_reconstructed = self.first_1x1conv(features)
        x_reconstructed = x_reconstructed + self.residual_conv(x_reconstructed)
        x_reconstructed = self.decoder(features)

        return x_reconstructed


class Vae(torch.nn.Module):
    def __init__(self, latent_size=128, fc_layers=(512, 512), num_residual_convs=3, device=None):
        super(Vae, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        self.encoder = VaeEncoder(latent_size, fc_layers, num_residual_convs, device=self.device)
        self.decoder = VaeDecoder(latent_size, fc_layers, num_residual_convs, device=self.device)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_sigma, z

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_size, device=self.device)
        fake_shmooots = self.decoder(z)

        return fake_shmooots
