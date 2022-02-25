import torch
from torch import functional as F


def vae_loss(x, x_reconstructed, z_mean, z_log_var, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum') / x.size(0)
    latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.size(0)

    return recon_loss + beta * latent_loss
