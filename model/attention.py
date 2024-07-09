import torch
from torch import nn
from torch.nn import functional
import math


class SelfAttention(nn.Module):
    """
    Get relations between tokens of an input sequence
    """

    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ) -> None:
        """
        Args:
        n_heads: number of heads on which attention mechanism will be splitted
        d_embed: size of the input embedding, here amount of features
        """
        # W_Q, W_K, W_V (key, query, value) matrices represented as one big matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # W_O output matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads

        # Amount of features to which each head attends
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        """
        Args:
        x: input (batch_size, seq_len, d_embed)
        """
        # Extract shape of the input
        batch_size, sequence_lenght, d_embed = x.shape

        # Multiply the input with the W_Q, W_K, W_V matrices
        # in:  (batch_size, seq_len, d_embed)
        # out: 3 * (batch_size, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Split Q, K, V matrices into the number of heads
        # Each head attends to just a part of the features
        # in:  (batch_size, seq_len, d_embed)
        # out: (batch_size, n_heads, seq_len, d_embed/n_heads)
        q = q.view((batch_size, sequence_lenght, self.n_heads, self.d_head)).transpose(
            1, 2
        )
        k = k.view((batch_size, sequence_lenght, self.n_heads, self.d_head)).transpose(
            1, 2
        )
        v = v.view((batch_size, sequence_lenght, self.n_heads, self.d_head)).transpose(
            1, 2
        )

        # Calculate attention
        attention_scores = q @ k.transpose(-1, -2)

        # Apply mask
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal)
            # is made up of 1 (useful for masking future tokens in NLP)
            mask = torch.ones_like(attention_scores, dtype=torch.bool).triu(1)
            attention_scores.masked_fill_(mask, -torch.inf)

        attention_scores /= math.sqrt(self.d_head)

        attention_scores = functional.softmax(attention_scores, dim=-1)

        # in:  (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_embed)
        # out: (batch_size, n_heads, seq_len, d_embed/n_heads)
        attention = attention_scores @ v

        # in:  (batch_size, n_heads, seq_len, d_embed/n_heads)
        # out: (batch_size, seq_len, d_embed)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(
            (batch_size, sequence_lenght, self.n_heads, self.d_head)
        )

        # Multiply attention with output matrix
        # in:  (batch_size, seq_len, d_embed)
        # out: (batch_size, seq_len, d_embed)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    """
    Calculate cross attention between latent variable and prompt signal
    """

    def __init__(
        self,
        n_heads: int,
        d_latent: int,
        d_context: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ) -> None:
        """
        Args:
        d_latent corresponds to the dimensionality of the latent variable
        d_context corresponds to the dimensionality of the prompt embedding
        """
        super().__init__()
        # Define W_q, W_k, W_v and W_o matrices
        self.q_proj = nn.Linear(d_latent, d_latent, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, d_latent, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, d_latent, bias=in_proj_bias)
        self.o_proj = nn.Linear(d_latent, d_latent, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_latent // n_heads

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
        latent: image features in latent space (batch_size, seq_len_latent, d_latent)
        context: prompt embeddings (batch_size, seq_len_context, d_context)
        """
        batch_size, seq_lenght_latent, d_latent = latent.shape

        # Multiply by matrices
        q = (
            self.q_proj(latent)
            .view((batch_size, -1, self.n_heads, self.d_head))
            .transpose(1, 2)
        )
        k = (
            self.k_proj(context)
            .view((batch_size, -1, self.n_heads, self.d_head))
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context)
            .view((batch_size, -1, self.n_heads, self.d_head))
            .transpose(1, 2)
        )

        attention_scores = q @ k.transpose(-1, -2)
        attention_scores /= math.sqrt(self.d_head)
        attention_scores = functional.softmax(attention_scores)
        attention = attention_scores @ v
        attention = attention.transpose(1, 2).continuous()
        attention = attention.view((batch_size, seq_lenght_latent, d_latent))

        return attention
