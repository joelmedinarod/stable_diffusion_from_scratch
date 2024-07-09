import torch
from torch import nn
from torch.nn import functional
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Normalization avoids that the outputs of a layer oscillate
        # too much. this helps to reduce training time (keep loss stable).
        # Each layer outputs values with mean 0 and variance 1
        # Group Normalization normalizes over rows/data items
        # (simular to layer normalization) but in groups
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: input (batch_size, feature channels, height, width)
        """

        residue = x

        batch_size, channels, height, width = x.shape

        # Transform shape of input image
        # in:  (batch_size, features, height, width)
        # out: (batch_size, features, height * width)
        x = x.view(batch_size, channels, height * width)

        # Transform shape for attention to relate
        # pixels to each other
        # in:  (batch_size, features, height * width)
        # out: (batch_size, height * width, features)
        x.transpose(-1, -2)

        # Calculate self attention
        x = self.attention(x)

        # Transform image back to original shape
        # in:  (batch_size, height * width, features)
        # out: (batch_size, features, height * width)
        x.transpose(-1, -2)

        # Transform shape of input image back to original
        # in:  (batch_size, features, height * width)
        # out: (batch_size, features, height, width)
        x = x.view(batch_size, channels, height, width)

        # Add residual connection
        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    """
    Block consiting of 2 normalization and 2 convolutional layers
    with a residual connection
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Add residual connection
        self.residual_layer == nn.Identity()
        if in_channels != out_channels:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: input (batch_size, in_channels, height, weight)
        """
        residue = x
        x = self.groupnorm_1(x)
        # non linear activation function
        x = functional.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = functional.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    """
    Generate image: Reduce number of channels and generate
    image with original shape
    """

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # in:  (batch_size, 512 feature channels, height/8, width/8)
            # out: (batch_size, 512 feature channels, height/4, width/4)
            nn.Upsample(scale_factor=2),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # in:  (batch_size, 512 feature channels, height/4, width/4)
            # out: (batch_size, 512 feature channels, height/2, width/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # Reduce number of features from 512 to 256
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # in:  (batch_size, 256 feature channels, height/2, width/2)
            # out: (batch_size, 256 feature channels, height, width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # Reduce number of features from 256 to 128
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # Normalize in groups of 32
            nn.GroupNorm(32, 128),
            # non linear activation function
            nn.SiLU(),
            # Generate image of original dimensions
            # in:  (batch_size, 128 feature channels, height, width)
            # out: (batch_size, 3 color channels, height, width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> None:
        """
        Args:
        x: input (batch_size, 4, height/8, width/8)
        """
        x /= 0.182215

        # Forward computation through model
        # out: (batch_size, 3, height, width)
        for module in self:
            x = module(x)

        return x
