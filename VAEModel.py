import torch
from torch import nn


def cond_batchnorm2d_layer(batch_norm, dims):
    if batch_norm:
        return nn.BatchNorm2d(dims)
    else:
        return nn.Identity()


def cond_batchnorm1d_layer(batch_norm, dims):
    if batch_norm:
        return nn.BatchNorm1d(dims)
    else:
        return nn.Identity()


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, batchnorm=False):
        super(ResidualConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.activation1 = nn.LeakyReLU()
        self.norm1 = nn.Identity() #nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same')
        self.activation2 = nn.Identity() #nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.activation2(out)
        out = self.norm2(out)

        return out + x


class VaeEncoder(nn.Module):
    def __init__(self, latent_size=128, fc_layers=(128, 128, 128), res_blocks_per_size=3,
                 device=None):
        super(VaeEncoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        feature_extractor_layers = [
            nn.Conv2d(3, 8, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
        ]

        for _ in range(res_blocks_per_size):
            feature_extractor_layers.append(ResidualConvBlock(8))
        feature_extractor_layers.append(nn.PixelUnshuffle(2))  # -> 32x64x64
        feature_extractor_layers.append(nn.Conv2d(32, 16, kernel_size=1, padding='same'))  # -> 16x64x64
        # feature_extractor_layers.append(nn.InstanceNorm2d(16))

        for _ in range(res_blocks_per_size):
            feature_extractor_layers.append(ResidualConvBlock(16))
        feature_extractor_layers.append(nn.PixelUnshuffle(2))  # -> 64x32x32
        feature_extractor_layers.append(nn.Conv2d(64, 32, kernel_size=1, padding='same'))  # -> 32x32x32
        # feature_extractor_layers.append(nn.InstanceNorm2d(32))

        for _ in range(res_blocks_per_size):
            feature_extractor_layers.append(ResidualConvBlock(32))
        feature_extractor_layers.append(nn.PixelUnshuffle(2))  # -> 128x16x16
        feature_extractor_layers.append(nn.Conv2d(128, 64, kernel_size=1, padding='same'))  # -> 64x16x16
        # feature_extractor_layers.append(nn.InstanceNorm2d(64))

        for _ in range(res_blocks_per_size):
            feature_extractor_layers.append(ResidualConvBlock(64))
        feature_extractor_layers.append(nn.Conv2d(64, 32, kernel_size=1, padding='same'))  # -> 32x16x16
        feature_extractor_layers.append(nn.LeakyReLU())
        # feature_extractor_layers.append(nn.InstanceNorm2d(32))
        feature_extractor_layers.append(nn.Conv2d(32, 16, kernel_size=1, padding='same'))  # -> 16x16x16
        feature_extractor_layers.append(nn.LeakyReLU())
        # feature_extractor_layers.append(nn.InstanceNorm2d(16))
        feature_extractor_layers.append(nn.Conv2d(16, 8, kernel_size=1, padding='same'))  # -> 8x16x16
        # feature_extractor_layers.append(nn.InstanceNorm2d(8))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        shared_fc_layers = [nn.Linear(8 * 16 * 16, fc_layers[0])]  # shared for mu and sigma
        for i in range(1, len(fc_layers)):
            shared_fc_layers.append(nn.Linear(fc_layers[i - 1], fc_layers[i]))
            shared_fc_layers.append(nn.LeakyReLU())
            shared_fc_layers.append(nn.LayerNorm(fc_layers[i]))

        self.shared_fc_layers = nn.Sequential(*shared_fc_layers)

        self.mu_fc = nn.Sequential(nn.Linear(fc_layers[-1], latent_size),
                                   nn.LeakyReLU(),
                                   nn.Linear(latent_size, latent_size))

        self.log_sigma_fc = nn.Sequential(nn.Linear(fc_layers[-1], latent_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(latent_size, latent_size))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.shared_fc_layers(features)
        mu = self.mu_fc(features)
        log_sigma = self.log_sigma_fc(features)

        # sample from normal distribution and reparameterize:
        z = torch.rand_like(mu)
        z = z * torch.exp(0.5 * log_sigma) + mu

        return z, mu, log_sigma


class VaeDecoder(torch.nn.Module):
    def __init__(self, latent_size=128, fc_layers=(128, 128, 128), res_blocks_per_size=3, device=None):
        super(VaeDecoder, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        # additional layers for the decoder to make it compatible with the encoder (shared fc layers)
        self.feature_extractor = [nn.Linear(latent_size, latent_size),
                                  nn.LeakyReLU(),
                                  nn.Linear(latent_size, latent_size),
                                  nn.LeakyReLU(),
                                  nn.Linear(latent_size, fc_layers[0])]
        for i in range(1, len(fc_layers)):
            self.feature_extractor.append(nn.Linear(fc_layers[i - 1], fc_layers[i]))
            self.feature_extractor.append(nn.LeakyReLU())
            self.feature_extractor.append(nn.LayerNorm(fc_layers[i]))
        self.feature_extractor.append(nn.Linear(fc_layers[-1], 8 * 16 * 16))
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        decoder_layers = [nn.Conv2d(8, 16, kernel_size=1, padding='same'),
                          nn.LeakyReLU(),
                          # nn.InstanceNorm2d(16),
                          nn.Conv2d(16, 32, kernel_size=1, padding='same'),
                          nn.LeakyReLU(),
                          # nn.InstanceNorm2d(32),
                          nn.Conv2d(32, 64,  kernel_size=1, padding='same')]  # -> 64x16x16

        for _ in range(res_blocks_per_size):
            decoder_layers.append(ResidualConvBlock(64))
        decoder_layers.append(nn.Conv2d(64, 128, kernel_size=1, padding='same'))  # -> 128x16x16
        decoder_layers.append(nn.PixelShuffle(2))  # -> 32x32x32
        # decoder_layers.append(nn.InstanceNorm2d(32))

        for _ in range(res_blocks_per_size):
            decoder_layers.append(ResidualConvBlock(32))
        decoder_layers.append(nn.Conv2d(32, 64, kernel_size=1, padding='same'))  # -> 64x32x32
        decoder_layers.append(nn.PixelShuffle(2))  # -> 16x64x64
        # decoder_layers.append(nn.InstanceNorm2d(16))

        for _ in range(res_blocks_per_size):
            decoder_layers.append(ResidualConvBlock(16))
        decoder_layers.append(nn.Conv2d(16, 32, kernel_size=1, padding='same'))  # -> 32x64x64
        decoder_layers.append(nn.PixelShuffle(2))  # -> 8x128x128
        # decoder_layers.append(nn.InstanceNorm2d(8))

        for _ in range(res_blocks_per_size):
            decoder_layers.append(ResidualConvBlock(8))
        decoder_layers.append(nn.Conv2d(8, 3, kernel_size=1, padding='same'))  # -> 3x128x128
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        features = self.feature_extractor(z)
        features = features.view(features.size(0), 8, 16, 16)
        x_reconstructed = self.decoder(features)

        return x_reconstructed


class Vae(torch.nn.Module):
    def __init__(self, latent_size=128, fc_layers=(128, 128), res_blocks_per_size=3, device=None):
        super(Vae, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_size = latent_size

        self.encoder = VaeEncoder(latent_size=latent_size,
                                  fc_layers=fc_layers,
                                  res_blocks_per_size=res_blocks_per_size,
                                  device=self.device)
        self.decoder = VaeDecoder(latent_size=latent_size,
                                  fc_layers=fc_layers,
                                  res_blocks_per_size=res_blocks_per_size,
                                  device=self.device)

    def forward(self, x):
        z, mu, log_sigma = self.encoder(x)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, mu, log_sigma, z

    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_size, device=self.device)
        fake_shmooots = self.decoder(z)

        return fake_shmooots
