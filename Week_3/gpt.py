import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# ------------------------- MaskedAttention -----------------------------
# This class implements multi-head self-attention with a causal mask. It ensures
# that each token can only attend to previous tokens (including itself) by masking future tokens.
class MaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Check that the embedding dimension is divisible by the number of heads.
        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Scale factor for attention logits.
        self.scale = self.head_dim**-0.5

        # Linear projections for keys, queries, values and the output.
        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Computes the multi-head self-attention of x with causal masking.
        Input:
            x: Tensor of shape [batch_size, seq_length, embed_dim]
        Output:
            Projected output with the same shape after applying attention.
        """

        batch_size, seq_length, embed_dim = x.size()
        # Compute keys, queries and values using linear projections.
        keys = self.k_projection(x)
        queries = self.q_projection(x)
        values = self.v_projection(x)

        # Rearrange keys, queries and values from shape [batch_size, seq_length, embed_dim]
        # to shape [(batch_size * num_heads), seq_length, head_dim] for efficient computation.
        keys = rearrange(
            keys, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )
        queries = rearrange(
            queries, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )
        values = rearrange(
            values, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )

        ####################### insert code here #########################################
        # Compute raw attention logits by matrix multiplication between queries and keys transpose,
        # and scale them using self.scale. This produces a pairwise similarity score for each query-key pair.
        attention_logits = torch.matmul(queries, keys.transpose(-2, -1))
        attention_logits = attention_logits * self.scale

        # Create a causal mask that only allows attention to tokens up to the current token.
        # The mask is upper-triangular (with zeros on the diagonal) to block future tokens.
        # torch.triu() can be used to generate such a mask.
        mask = torch.triu(torch.ones_like(attention_logits), diagonal=1).bool()

        # Use torch.masked_fill() to expand the mask and fill masked positions with a very negative value.
        # This prevents the attention mechanism from considering future tokens.
        attention_logits = attention_logits.masked_fill(mask, float('-inf'))

        # Compute attention probabilities by applying softmax on the attention logits.
        # Then compute the weighted sum of the value vectors using these probabilities.
        attention = torch.softmax(attention_logits, dim=-1)
        out = torch.bmm(attention, values)
        ###################################################################################

        # Rearrange the output back from the shape [(batch_size * num_heads), seq_length, head_dim]
        # to [batch_size, seq_length, embed_dim] by concatenating the heads.
        out = rearrange(
            out, "(b h) s d -> b s (h d)", h=self.num_heads, d=self.head_dim
        )

        # Check that the attention weights and final output have the expected shapes.
        assert attention.size() == (batch_size * self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        # Final linear projection of the outputs.
        return self.o_projection(out)


# ------------------------- EncoderBlock -----------------------------
# This block represents one transformer encoder layer composed of a masked self-attention
# mechanism followed by a feed-forward network, each with layer normalization and dropout.
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()
        # Layer normalization before self-attention.
        self.layernorm1 = nn.LayerNorm(embed_dim)
        # Layer normalization before the feed-forward network.
        self.layernorm2 = nn.LayerNorm(embed_dim)
        # The masked self-attention layer.
        self.attention = MaskedAttention(embed_dim=embed_dim, num_heads=num_heads)
        # Set the hidden dimension of the feed-forward network.
        fc_hidden_dim = 4 * embed_dim if fc_dim is None else fc_dim
        # Feed-forward network with a GELU nonlinearity.
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ----- Self-Attention Block -----
        # Store the residual connection.
        residual = x
        # Apply layer normalization before the attention as a pre-processing step.
        x = self.layernorm1(x)
        # Compute attention outputs.
        attention_out = self.attention(x)
        # Add the attention outputs to the residual.
        x = residual + attention_out
        # Apply dropout to the result.
        x = self.dropout(x)

        # ----- Feed-Forward Block -----
        # Add residual connection from attention block.
        residual = x
        # Apply layer normalization before the feed-forward network.
        x = self.layernorm2(x)
        # Compute the feed-forward network output.
        fc_out = self.fc(x)
        # Add the feed-forward outputs to the residual.
        x = residual + fc_out
        # Apply dropout.
        x = self.dropout(x)
        return x


# ------------------------- Positional Encoding -----------------------------
# Implements fixed sinusoidal positional encoding to provide tokens with
# information about their positions in the sequence.
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        # Initialize a tensor for positional encodings with zeros.
        pe = torch.zeros(max_seq_len, embed_dim)
        # Create a vector of positions (0 to max_seq_len-1).
        position = torch.arange(0.0, max_seq_len).unsqueeze(1)
        # Compute a divisor term based on the embedding dimension.
        div_term = torch.exp(
            torch.arange(0.0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        # Apply sine to even indices in the positional encoding.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices.
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add an extra batch dimension.
        pe = pe.unsqueeze(0)
        # Register the positional encoding as a non-trainable buffer.
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        # Add the positional encoding to the input token embeddings.
        return x + self.pe[:, :seq_length]


# ------------------------- Positional Embedding -----------------------------
# Implements learned positional embeddings using nn.Embedding.
class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEmbedding, self).__init__()
        # The embedding layer for positions.
        self.pe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        # Generate position indices using torch.arange.
        positions = self.pe(
            torch.arange(
                seq_length, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        # Expand the positions to match the batch size.
        positions = positions[None, :, :].expand(batch_size, seq_length, embed_dim)
        # Add the learned positional embeddings to the input token embeddings.
        return x + positions


# ------------------------- AndersenGPT -----------------------------
# Implements the GPT-style transformer model for next-token prediction.
class AndersenGPT(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        pos_enc="fixed",
        dropout=0.0,
        fc_dim=None,
        num_tokens=50_000,
    ):
        """
        Initializes the GPT model.
        Parameters:
            embed_dim: Dimensionality of token embeddings.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            max_seq_len: Maximum sequence length for the model.
            pos_enc: Type of positional encoding ("fixed" for sinusoidal or "learnable").
            dropout: Dropout rate.
            fc_dim: Hidden dimension in the feed-forward network (default: 4 * embed_dim).
            num_tokens: Number of tokens in the vocabulary.
        """
        super().__init__()

        # Token embedding layer.
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        # Choose the type of positional encoding.
        if pos_enc == "fixed":
            self.positional_encoding = PositionalEncoding(
                embed_dim=embed_dim, max_seq_len=max_seq_len
            )
        elif pos_enc == "learnable":
            self.positional_encoding = PositionalEmbedding(
                embed_dim=embed_dim, max_seq_len=max_seq_len
            )

        # Build a stack of transformer encoder blocks.
        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    fc_dim=fc_dim,
                    dropout=dropout,
                )
            )
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        # Final layer normalization.
        self.ln_f = nn.LayerNorm(embed_dim)

        # Linear layer to project hidden states to vocabulary logits.
        self.lm_head = nn.Linear(embed_dim, num_tokens, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the GPT model.
        Parameters:
            x: Tensor of token indices with shape [batch_size, seq_length].
        Returns:
            Logits of shape [batch_size, seq_length, num_tokens] for next token prediction.
        """
        # Get token embeddings.
        tokens = self.token_embedding(x)
        # Add positional encoding to token embeddings.
        tokens = self.positional_encoding(tokens)
        # Apply dropout to the embeddings.
        tokens = self.dropout(tokens)
        # Pass through the transformer blocks.
        x = self.transformer_blocks(tokens)
        # Apply final layer normalization.
        x = self.ln_f(x)
        # Compute logits for each token.
        logits = self.lm_head(x)
        return logits
