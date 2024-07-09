import torch
from torch import nn
from torch.nn import functional
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, d_embed: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, 4 * d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform time embedding (1, 320) -> (1, 1280)"""
        x = self.linear_1(x)
        x = functional.silu(x)
        x = self.linear_2(x)
        return x


class SwitchSequential(nn.Sequential):
    """
    Use different arguments during the forward pass
    depending on the type of layer the data is going through
    """

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNETAttentionBlock):
                # compute cross attention
                x = layer(x, context)
            elif isinstance(layer, UNETResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):

    def __init__(self, channels: int) -> None:

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, features, height, width) -> (batch_size, features, height*2, width*2)
        x = functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNETResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, d_time: int = 1280) -> None:
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # Feed Forward Layer for the time embedding
        self.linear_time = nn.Linear(d_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        features: latent variables (batch_size, inchannel, height, width)
        time: time embedding (1, 1280)
        """
        residue = features

        # Transform image features
        features = self.groupnorm_feature(features)
        features = functional.silu(features)
        features = self.conv_feature(features)

        # Transform time embedding
        time = functional.silu(time)
        time = self.linear_time(time)

        # Merge time and features embeddings, and transform them
        merged = features + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = functional.silu(merged)
        merged = self.conv_merged(merged)

        # Add residual connection
        return merged + self.residual_layer(residue)


class UNETAttentionBlock(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_context: int = 768) -> None:
        super().__init__()
        channels = n_heads * d_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads, channels, d_context, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> None:
        """
        Args:
        x: input, latent variable (batch_size, features, height, width)
        context: prompt (batch_size, seq_len, d_embed)
        """

        # this residual connection will be applied at the end
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        # Take shape of the input
        batch_size, channels, height, width = x.shape

        # Modify input shape for attention
        # (batch_size, features, height, width) -> (batch_size, height*width, features)
        x = x.view((batch_size, channels, height * width))
        x = x.transpose(-1, -2)

        # Normalization + SelfAttention with residual connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization + CrossAttention with residual connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # Feed Forward Layer with Geglu activation function
        # and residual connection
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * functional.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        # Modify tensor back to image dimensions
        # (batch_size, height*width, features) -> (batch_size, features, height, width)
        x = x.transpose(-1, -2)
        x = x.view((batch_size, channels, height, width))

        return self.conv_output(x)


class UNET(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/16, width/16)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # ResidualBlock + Attention with 8 heads on 40 features
                SwitchSequential(
                    UNETResidualBlock(320, 320), UNETAttentionBlock(8, 40)
                ),
                # (batch_size, 320, height/8, width/8) -> (batch_size, 320, height/16, width/16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                # (batch_size, 320, height/16, width/16) -> (batch_size, 640, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(320, 640), UNETAttentionBlock(8, 80)
                ),
                # ResidualBlock + Attention with 8 heads on 80 features
                SwitchSequential(
                    UNETResidualBlock(640, 640), UNETAttentionBlock(8, 80)
                ),
                # (batch_size, 640, height/16, width/16) -> (batch_size, 640, height/32, width/32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                # (batch_size, 640, height/32, width/32) -> (batch_size, 1280, height/32, width/32)
                SwitchSequential(
                    UNETResidualBlock(640, 1280), UNETAttentionBlock(8, 160)
                ),
                # ResidualBlock + Attention with 8 heads on 160 features
                SwitchSequential(
                    UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)
                ),
                # (batch_size, 640, height/32, width/32) -> (batch_size, 640, height/64, width/64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                # (batch_size, 640, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
                SwitchSequential(
                    UNETResidualBlock(1280, 1280), UNETAttentionBlock(8, 160)
                ),
                # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
                SwitchSequential(UNETResidualBlock(1280, 1280)),
                # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
                SwitchSequential(UNETResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNETResidualBlock(1280, 1280),
            UNETAttentionBlock(8, 160),
            UNETResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # Take output of the bottleneck and skip connection from encoder
                # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
                SwitchSequential(UNETResidualBlock(2560, 1280)),
                # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
                SwitchSequential(UNETResidualBlock(2560, 1280)),
                # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/32, width/32)
                SwitchSequential(UNETResidualBlock(2560, 1280), UpSample(1280)),
                # (batch_size, 2560, height/32, width/32) -> (batch_size, 1280, height/32, width/32)
                SwitchSequential(
                    UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)
                ),
                # (batch_size, 2560, height/32, width/32) -> (batch_size, 1280, height/32, width/32)
                SwitchSequential(
                    UNETResidualBlock(2560, 1280), UNETAttentionBlock(8, 160)
                ),
                # (batch_size, 1920, height/32, width/32) -> (batch_size, 1280, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(1920, 1280),
                    UNETAttentionBlock(8, 160),
                    UpSample(1280),
                ),
                # (batch_size, 1920, height/16, width/16) -> (batch_size, 640, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(1920, 640), UNETAttentionBlock(8, 80)
                ),
                # (batch_size, 1280, height/16, width/16) -> (batch_size, 640, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(1280, 640), UNETAttentionBlock(8, 80)
                ),
                # (batch_size, 960, height/16, width/16) -> (batch_size, 640, height/8, width/8)
                SwitchSequential(
                    UNETResidualBlock(960, 640),
                    UNETAttentionBlock(8, 80),
                    UpSample(640),
                ),
                # (batch_size, 960, height/8, width/8) -> (batch_size, 320, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(960, 320), UNETAttentionBlock(8, 40)
                ),
                # (batch_size, 640, height/8, width/8) -> (batch_size, 320, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)
                ),
                # (batch_size, 640, height/8, width/8) -> (batch_size, 320, height/16, width/16)
                SwitchSequential(
                    UNETResidualBlock(640, 320), UNETAttentionBlock(8, 40)
                ),
            ]
        )


class UNETOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 320 features into 4 features"""
        x = self.groupnorm(x)
        x = functional.silu(x)
        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    """
    U-Net

    Receive noisy image, noisification time step and prompt,
    and generate an image in the latent space of the variational
    autoencoder
    """

    def __init__(self) -> None:
        # size of the time embedding: 320
        # Model receives embedding of timestep
        # at which image was noisified
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNETOutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        latent: output of the encoder of the variational autoencoder
            (batch_size, 4, height/8, width/8)
        context: prompt converted by the CLIP Encoder
            (batch_size, seq_len, d_embed)
        time: embedding of time tensor (1, 320)
        """
        # similar to positional encoding of the transformer to
        # tell the model information about denoisification step
        # in:  (1, 320)
        # out: (1, 1280)
        time = self.time_embedding(time)

        # Forward pass through UNET. Get 320 output features
        # out of 4 input features in latent space
        # in:  (batch_size, 4, height/8, width/8)
        # out: (batch_size, 320, height/8, width/8)
        unet_output = self.unet(latent, context, time)

        # Get new variable in latent space of variational autoencoder
        output = self.output_layer(unet_output)
