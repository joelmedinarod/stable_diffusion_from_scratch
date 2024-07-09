import torch
from torch import nn
from torch.nn import functional
from attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_embed: int, seq_len: int):
        """
        Args:
        vocab_size: amount of different tokens in the vocabulary
        d_embed: dimensionality of the embeddings
        seq_len: maximal lenght of a sequence
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        # parameters for positional encoding will be learned by the model
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, d_embed))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
        tokens: (batch_size, seq_len, d_embed)
        """
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, d_embed: int) -> None:
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x: input (batch_size, seq_len, d_embed)
        """
        residue = x

        # Masked Self Attention (In a sentence, ignore future tokens)
        # with residual connection
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        # Feed Forward Layer with residual connection
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self):
        # Vocabulary Size: 49408
        # Dimensionality of Embedding: 768
        # Maximal sequence lenght: 77
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList(
            [
                # Number of Heads of Multi-Head-Attention: 12
                # Dimensionality of the Embedding: 768
                # Number of CLIPLayers: 12
                CLIPLayer(12, 768)
                for i in range(12)
            ]
        )

        # Perform Layer Normalization
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # in:  (batch_size, seq_len)
        # out: (batch_size, seq_len, d_embed)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        return self.layer_norm(state)
