import torch

from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class VAEConfig:
    image_channels: int = 3
    image_size: int = 64
    init_channels: int = 64
    latent_dim: int = 128
    scaling_factor: int = 255
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        c = config.init_channels
        self.feature_size = config.image_size // 16
        self.feature_dim = 8 * c * self.feature_size * self.feature_size

        def down(in_channels, out_channels):
            return [nn.Conv2d(in_channels, out_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()]

        def up(in_channels, out_channels):
            return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()]

        self.encoder = nn.Sequential(
            *down(config.image_channels, c), # (batch_size, c, 32, 32)
            *down(c, 2 * c), # (batch_size, 2c, 16, 16)
            *down(2 * c, 4 * c), # (batch_size, 4c, 8, 8)
            *down(4 * c, 8 * c), # (batch_size, 8c, 4, 4)
            nn.Flatten(),
            nn.Linear(self.feature_dim, 2 * config.latent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, self.feature_dim),
            nn.Unflatten(1, (8 * c, self.feature_size, self.feature_size)), # (batch_size, 8c, 4, 4)
            nn.BatchNorm2d(8 * c),
            nn.GELU(),
            *up(8 * c, 4 * c), # (batch_size, 4c, 8, 8)
            *up(4 * c, 2 * c), # (batch_size, 2c, 16, 16)
            *up(2 * c, c), # (batch_size, c, 32, 32)
            nn.ConvTranspose2d(c, config.image_channels, kernel_size=config.kernel_size, stride=config.stride, padding=config.padding), # (batch_size, 3, 64, 64)
            nn.Sigmoid())

    def encode(self, x):
        x = self.encoder(x) # (batch_size, 2 * latent_dim)
        return x.split(self.config.latent_dim, dim=-1) # (batch_size, latent_dim), (batch_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x_pred, x, mu, logvar):
        """Computes negative ELBO"""
        recon = F.binary_cross_entropy(x_pred, x, reduction="sum") # returns nll
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        nelbo = recon + kld
        return nelbo

    def scale_up(self, x_pred):
        return x_pred * self.config.scaling_factor

    def forward(self, x):
        # x.shape = (batch_size, 3, 64, 64)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z) # (batch_size, 3, 64, 64)

        return self.scale_up(x_pred), self.loss(x_pred, x, mu, logvar)

    def generate(self, num_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.config.latent_dim, device=device) # (num_samples, latent_dim)
        return self.decode(z) # (num_samples, 3, 64, 64)


@dataclass
class SimpleVAEConfig:
    image_channels: int = 3
    image_size: int = 64
    input_dim: int = image_channels * image_size * image_size
    hidden_dim: int = 512
    latent_dim: int = 32
    scaling_factor: int = 255

class SimpleVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2 * config.latent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid())

    def encode(self, x):
        x = self.encoder(x) # (batch_size, 2 * latent_dim)
        return x.split(self.config.latent_dim, dim=-1) # (batch_size, latent_dim), (batch_size, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x_pred, x, mu, logvar):
        """Computes negative ELBO"""
        recon = F.binary_cross_entropy(x_pred, x, reduction="sum") # returns nll
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        nelbo = recon + kld
        return nelbo

    def scale_up(self, x_pred):
        return x_pred * self.config.scaling_factor

    def forward(self, x):
        B, C, H, W = x.shape # (batch_size, 3, 64, 64)
        x = x.reshape(B, C*H*W) # (batch_size, input_dim)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decode(z) # (batch_size, input_dim)
        
        return self.scale_up(x_pred), self.loss(x_pred, x, mu, logvar)

    def generate(self, num_samples=1):
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.config.latent_dim, device=device) # (num_samples, latent_dim)
        return self.decode(z) # (num_samples, input_dim)
