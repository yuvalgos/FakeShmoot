import torch
import torch.nn.functional as F
import torchvision


def vae_loss(x, x_reconstructed, z_mean, z_log_var, beta=1.0):
    """ Original VAE loss function  composed of reconstruction loss and KL divergence """
    recon_loss = torch.sum((x - x_reconstructed) ** 2) / x.numel()
    latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.numel()

    loss = recon_loss + beta * latent_loss
    return loss, recon_loss, latent_loss


class VGGPerceptualLoss(torch.nn.Module):
    """ Copied entirely from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49 """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class VAEComposedLoss(torch.nn.Module):
    def __init__(self, kl_beta=1.5, vgg_weight=0.00, VGG_feature_layers=[0, 1, 2, 3],
                 VGG_style_layers=[], device='auto'):
        super(VAEComposedLoss, self).__init__()

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.kl_beta = kl_beta
        self.vgg_weight = vgg_weight
        self.feature_layers = VGG_feature_layers
        self.style_layers = VGG_style_layers
        self.vgg_loss = VGGPerceptualLoss(resize=False,).to(self.device)

    def forward(self, x, x_reconstructed, z_mean, z_log_var):
        _, recon_mse_loss, latent_loss = vae_loss(x, x_reconstructed, z_mean, z_log_var, beta=self.kl_beta)
        vgg_loss = self.vgg_loss(x, x_reconstructed, feature_layers=self.feature_layers,
                                 style_layers=self.style_layers)

        recon_loss = recon_mse_loss + self.vgg_weight * vgg_loss
        loss = recon_loss + self.kl_beta * latent_loss
        return loss, vgg_loss, recon_mse_loss, latent_loss

