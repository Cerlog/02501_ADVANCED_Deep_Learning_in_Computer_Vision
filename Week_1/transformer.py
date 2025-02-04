import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        print("Constructing Attention Layer")
        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        print("*"*50)
        print(f"Embedding dimension: {embed_dim}, Number of heads: {num_heads}")
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        print(f"Scaled by: {self.scale}")
        print("*"*50)
        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        print(f"Keys projection: {embed_dim} x {embed_dim}")
        print(f"Keys projection:{self.k_projection}")
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        print(f"Queries projection: {embed_dim} x {embed_dim}")
        print(f"Queries projection:{self.q_projection}")
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        print(f"Values projection: {embed_dim} x {embed_dim}")
        print(f"Values projection:{self.v_projeciton}")
        self.o_projection = nn.Linear(embed_dim, embed_dim)
        print(f"Output projection: {embed_dim} x {embed_dim}")
        print(f"Output projection:{self.o_projection}")

    def forward(self, x):
        print("*"*50)
        print("Foward pass through Attention Layer")

        """
        This function computes the multi-head self-attention of x.
        """
        batch_size, seq_length, embed_dim = x.size()
        print(f"Batch size: {batch_size}, Sequence length: {seq_length}, Embedding dimension: {embed_dim}")
        print(f"X shape: {x.shape}")
        print(f"X: {x}")
        print("*"*50)
        print("Generating keys, queries, and values")
        # Generate keys, queries, and values
        keys    = self.k_projection(x) # B x seq_len x embed_dim
        print(f"Keys shape: {keys.shape}")
        print(f"Keys: {keys}")
        queries = self.q_projection(x) # B x seq_len x embed_dim
        print(f"Queries shape: {queries.shape}")
        print(f"Queries: {queries}")
        values  = self.v_projeciton(x) # B x seq_len x embed_dim
        print(f"Values shape: {values.shape}")
        print(f"Values: {values}")
        print("*"*50)

        """
        Now you have to split the projected keys, queries, and values to multiple heads.
        """
        # First split the embed_dim to num_heads x head_dim
        print("*"*50)
        keys = rearrange(keys, 'b s (h d) -> b h s d', h=self.num_heads)
        print(f"Keys shape after splitting: {keys.shape}")
        print(f"Keys after splitting: {keys}")
        print("*"*50)
        # Secondly merge the batch_size with the num_heads
        print("Keys shape before merging batch_size with num_heads")
        keys = rearrange(keys, 'b h s d -> (b h) s d')
        print(f"Keys shape after merging batch_size with num_heads: {keys.shape}")
        # HINT repeat the same process for queries and values
        print("*"*50)
        print(f"Queries shape before splitting and merging {queries.shape}")
        queries = rearrange(queries, 'b s (h d) -> b h s d', h=self.num_heads)
        print(f"Queries shape after splitting: {queries.shape}")
        queries = rearrange(queries, 'b h s d -> (b h) s d')
        print(f"Queries shape after merging: {queries.shape}")
        print("*"*50)

        print(f"Values shape before splitting and merging {values.shape}")
        values = rearrange(values, 'b s (h d) -> b h s d', h=self.num_heads)
        print(f"Values shape after splitting: {values.shape}")
        values = rearrange(values, 'b h s d -> (b h) s d')
        print(f"Values shape after merging: {values.shape}")

        # Compute attetion logits
        print("*"*50)
        attention_logits = torch.bmm(queries, keys.transpose(1, 2))
        print(f"Attention logits shape: {attention_logits.shape}")
        print(f"Attention logits: {attention_logits}")
        print("*"*50)
        attention_logits = attention_logits * self.scale
        print(f"Attention logits after scaling: {attention_logits}")
        print(f"Attention logits shape after scaling: {attention_logits.shape}")
        print("*"*50)
        attention = F.softmax(attention_logits, dim=-1)
        print(f"Attention shape: {attention.shape}")
        print(f"Attention: {attention}")
        print("*"*50)

        # Apply attention to values
        print("Applying attention to values")
        print(f"Values shape and attention shape: {values.shape}, {attention.shape}")
        out = torch.bmm(attention, values)
        print(f"Output shape: {out.shape}")
        print(f"Output: {out}")
        print("*"*50)

        # Rearragne output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        
        # get half of the embedding indices
        div_term = torch.arange(0., embed_dim, 2)
        # miltiply each position with -(math.log(10000.0) / embed_dim)
        div_term = -(math.log(10000.0) / embed_dim) * div_term
        # compute the exp of div_term
        div_term = torch.exp(div_term)
        # Use slice notation with step=2 for even and odd positions
        pe[:, 0::2] = torch.sin(position * div_term)    # Even positions (0,2,4,...)
        pe[:, 1::2] = torch.cos(position * div_term)    # Odd positions (1,3,5,...)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(embedding_dim=embed_dim, num_embeddings=max_seq_len)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        positions = self.pe(torch.arange(seq_length, device=to_device()))
        positions = positions[None, :, :].expand(batch_size, seq_length, embed_dim)
        return x + positions
        

class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None, num_tokens=50_000, num_classes=2, 
                 
    ):
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']
        

        self.pool, self.pos_enc, = pool, pos_enc
        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)

        # Initialize cls token parameter
        if self.pool == 'cls':
            #self.cls_token = ... # Institate an nn.Parameter with size: 1,1,embed_dim
            max_seq_len +=1
        
        if self.pos_enc == 'fixed':
            self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif self.pos_enc == 'learnable':
            self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        tokens = self.token_embedding(x)
        batch_size, seq_length, embed_dim = tokens.size()

        # Include cls token in the input sequence
        ####################### insert code here #######################
        if self.pool == 'cls':
            # HINT: repeat the cls token of the batch dimension
            pass
        ################################################################

        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        x = self.transformer_blocks(x)

        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
            
        # Get cls token
        ####################### insert code here #######################
        elif self.pool == 'cls':
            # HINT: get the first output token of the transfomer.
            pass
        ################################################################

        return self.classifier(x)
    


def test_attention():
    torch.manual_seed(0)  # for reproducibility

    # Define dimensions
    batch_size = 2
    seq_length = 5
    embed_dim = 16
    num_heads = 4

    # Create a random input tensor: $$x \in \mathbb{R}^{B \times S \times E}$$
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Initialize the Attention module
    attention_layer = Attention(embed_dim=embed_dim, num_heads=num_heads)

    # Pass the input through the attention module
    output = attention_layer(x)

    # Print shapes to help you see what happens at each stage
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_attention()