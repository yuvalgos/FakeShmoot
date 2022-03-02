import torch
import torch.nn.functional as F


def vae_loss(x, x_reconstructed, z_mean, z_log_var, beta=1.0):
    recon_loss = torch.sum((x - x_reconstructed) ** 2) / x.numel()
    latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.numel()

    loss = recon_loss + beta * latent_loss
    return loss, recon_loss, latent_loss
