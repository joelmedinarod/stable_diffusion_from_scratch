"""
Encoder of Variational Autoencoder
Used to compress images and represent it as a variable in a latent space.
"""

import torch
from torch import nn
from torch.nn import functional
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """
    Reduce dimensionality of input images and increase number of features
    """

    def __init__(self):
        """
        Compress image and learn multivariate probability distribution
        representing a latent space.

        The encoder returns the parameters of the probability
        distribution: mean and log_variance of the data.
        """
        super().__init__(
            # Get features from input image
            # in:  (batch_size, 3 rgb channel, height, width)
            # out: (batch_size, 128 feature channels, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # Combines convolutions with normalization
            # in: (batch_size, 128 feature channels, height, width)
            # out: (batch_size, 128 feature channels, height, width)
            VAE_ResidualBlock(128, 128),
            # Get new features and reduce size of the image
            # in:  (batch_size, 128 feature channels, height, width)
            # out: (batch_size, 128 feature channels, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # Normalize and increase number of features
            # in:  (batch_size, 128 feature channels, height/2, width/2)
            # out: (batch_size, 256 feature channels, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            # in:  (batch_size, 256 feature channels, height/2, width/2)
            # out: (batch_size, 256 feature channels, height/2, width/2)
            VAE_ResidualBlock(256, 256),
            # Convolutions and reduce size of the image
            # in:  (batch_size, 256 feature channels, height/2, width/2)
            # out: (batch_size, 256 feature channels, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # Increase number of features
            # in:  (batch_size, 256 feature channels, height/4, width/4)
            # out: (batch_size, 512 feature channels, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            # in:  (batch_size, 256 feature channels, height/4, width/4)
            # out: (batch_size, 512 feature channels, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            # Convolutions and reduce size of the image
            # in:  (batch_size, 256 feature channels, height/4, width/4)
            # out: (batch_size, 256 feature channels, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 512 feature channels, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # Self-Attention to get relationships between all pixels from
            # image (global information)
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 512 feature channels, height/8, width/8)
            VAE_AttentionBlock(512),
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 512 feature channels, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            # Group Normalization with 32 groups and 512 feature channels
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 512 feature channels, height/8, width/8)
            nn.GroupNorm(32, 512),
            # Activation Function
            nn.SiLU(),
            # Bottleneck of the encoder: reduce the number of features
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 8 feature channels, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # in:  (batch_size, 8 feature channels, height/8, width/8)
            # out: (batch_size, 8 feature channels, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: input image (batch_size, in_channels=3, height, width)
        noise: added to encoder output (batch_size, out_channels, height/8, width/8)
        """
        # Run modules sequentially
        for module in self:
            # Add special padding to convolutions with stride = 2
            if getattr(module, "stride", None) == (2, 2):
                # Add layer of pixels just on the right and
                # bottom sides of the image
                x = functional.pad(x, (0, 1, 0, 1))
            x = module(x)

        # Get parameters of probability distribution of data
        # in:  (batch_size, 8 feature channels, height/8, weight/8)
        # out: 2 * (batch_size, 4 feature channels, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Define lower and upper limit to the log_variance
        log_variance = torch.clamp(log_variance, -30, 20)

        # Transform log variance to variance
        variance = log_variance.exp()

        # Calculate standard deviation
        std = variance.sqrt()

        # Sample from latent space by adding learned parameters
        # to the noise (normal distribution N(0, 1))
        # Z = N(0, 1) -> N(mean, variance): X = mean + std * Z
        x = mean + std * noise

        # Scale output by a constant (refer to paper)
        x *= 0.18215
