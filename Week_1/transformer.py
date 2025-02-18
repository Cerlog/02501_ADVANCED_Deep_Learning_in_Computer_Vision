import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat # A library to make tensor rearrangements more readable and less error prone then using view, permute, transpose. etc 

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

class Attention(nn.Module):
    """
    - Self-attention allows each position in a sequence to attend (i.e., focus) 
    to other positions in the same sequence. In Transformers, we do this by 
    computing a similarity-based weighting between “queries” and “keys” to 
    then combine the “values.”

    - Multi-head means we split the embedding dimension into multiple “heads,” 
    each learning potentially different relationships.
    
    
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # i.e 1/sqrt(head_dim)
        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        - This function computes the multi-head self-attention of x.
        - The input x has shape (batch_size, seq_length, embed_dim)
        - We learn separate linear transformations for Keys (k_projection), 
            Queries (q_projection), and Values (v_projection)
        - Mathematically, if X is the input: 
            - K = XW_k, Q = XW_q, V = XW_v
            - where W_k, W_q, W_v are the learnable weights for the projections 
            of dimensions R^{embed_dim x embed_dim}	
        """

        # 1) Project x into keys (K), queries (Q), and values (V)
        batch_size, seq_length, embed_dim = x.size()
        keys    = self.k_projection(x) # B x seq_len x embed_dim
        queries = self.q_projection(x) # B x seq_len x embed_dim
        values  = self.v_projeciton(x) # B x seq_len x embed_dim

        """
        Now you have to split the projected keys, queries, and values to multiple heads.
        After we get K, Q, V (each of them of shape B, S, embed_dim), we split them into 
        (Batch_size, num_heads, seq_length, head_dim) for a parallel computation. 
        Then we merge the batch dimension and the head dimension to get a parallel computation 
        for all the heads at once. 
        """
        # 2) Split each into multiple heads: (batch_size, seq_length, embed_dim) -> (batch_size, num_heads, seq_length, head_dim)
        # and then merge ((B * num_heads), seq_length, head_dim)
        keys = rearrange(keys, 'b s (h d) -> b h s d', h=self.num_heads)
        keys = rearrange(keys, 'b h s d -> (b h) s d')
        queries = rearrange(queries, 'b s (h d) -> b h s d', h=self.num_heads)
        queries = rearrange(queries, 'b h s d -> (b h) s d')

        values = rearrange(values, 'b s (h d) -> b h s d', h=self.num_heads)
        values = rearrange(values, 'b h s d -> (b h) s d')


        """
        Dot product: We compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V. 
        This is a "scaled dot product attention" where the attention weights are computed

        # Scaling: Dividing by sqrt(d_k) is done to prevent the logits from getting too large 
        and it helps to stabilize the gradients during training.

        Result: Finally, each head's output is concatenated (via rearrange and fed 
        through a final layer o_projection)

        Using multi-head attention increases the models ability to focus on different 
        positions or relations within the sequence.

        """


        # 3) Dot product QK^T to get attention logits
        attention_logits = torch.bmm(queries, keys.transpose(1, 2)) # (batch_size * num_heads, seq_length, seq_length)
        # 4) Scale by sqrt of head_dim (self.scale = 1 / sqrt(head_dim))
        attention_logits = attention_logits * self.scale
        # 5) Softmar along the last dimension to get attention weights
        attention = F.softmax(attention_logits, dim=-1)

        # Apply attention to values
        # 6) Multiply attention weigthts by values
        out = torch.bmm(attention, values) # shape: (batch_size * num_heads, seq_length, head_dim)

        # 7) Reshape back to (B, S, embed_dim)
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)


        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        # 8) Final linear projection 
        return self.o_projection(out)

class EncoderBlock(nn.Module):
    """
    The encoder block combines two crucial sub-layers:
    - Multi-head self-attention mechanism (from the Attention class)
    - Position-wise feed-forward networks (FFN)

    Both sub-layers are followed by a residual connection (skip connections)
    and Layer normalization. Typically, we also apply dropout to each sub-layers 
    output (after or before the addition with the input, depending on the implementation).

    Feed-Forward Network: Mathematical perspective 
    - It is often called "Position-wise FFN", because it is applied identically 
    to each position (token) in the sequence, without mixing token position in this stage. 

    In the original Transfoer, this sub-layer is: 

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2, where: 
    - W_1 \in R^{embed_dim x fc_dim}, b_1 \in R^{fc_dim} and W_2 \in R^{fc_dim x embed_dim}, b_2 \in R^{embed_dim}
    - ReLU acts element-wise, so each token is tranformed independently, but with the same weights. 

    ! Why 4*embed_dim in the first linear layer?
    - using a larger hidden dimension (often 2x or 4x the embedding size) works well, improving representational capacity.



    Residual Connection 
    A residual (or skip) connection adds the original input x back to the output of the sub-layer 
    x_out = LayerNorm(x + SubLayer(x_in))

    - Improves gradient flow, allowing deeper network to be trained 
    - Helps to avoid the vanishing gradient problem

    """

    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        # 1) Multi-head self-attention sub-layer
        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        
        # 2) LayerNorm after the attention sub-layer
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        # 3) Feed-forward sub layer
        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )

         # 4) LayerNorm after the feed-forward sub-layer 
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # 5) Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape (batch_size, seq_length, embed_dim)

        # -- Sub Layer 1 : Self-Attention
        attention_out = self.attention(x) # (batch_size, seq_length, embed_dim)
        # Residual connection + LayerNorm
        x = self.layer_norm1(attention_out + x)
        # Apply dropout
        x = self.dropout(x)

        # -- Sub Layer 2: Feed-Forward Network --
        fc_out = self.fc(x) # (batch_size, seq_length, embed_dim)
        # Residual connection + LayerNorm
        x = self.layer_norm2(fc_out + x)
        x = self.dropout(x)

        return x # (batch_size, seq_length, embed_dim)

class PositionalEncoding(nn.Module):
    """
    -Self attention is order-agnostic. To incorporate the order of the sequence, there are two main strategies:
    - Fixed positional encodings: Add a fixed set of sinusoidal functions of different frequencies to the input embeddings.
    - Learned positional encodings: Learn the positional encodings during training.


    # How it works mathematically 
    - For each position pos \in [0, ..., max_seq_len-1] and each dimension i \in [0, ..., embed_dim-1],
    - Even indices (i=2k) are computed using sin, and odd indices (i=2k+1) are computed using cos.
    Intuition: 
    - Low-frequency computents (for smaller k) vary slowly as pos changes. 
    - High frequency coponents (for larger k) vary more rapidly as pos changes.
    - The combination gives a rich representation of the position in the sequence.
    """


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

        pe = pe.unsqueeze(0) # shape (1, max_seq_len, embed_dim)
        self.register_buffer('pe', pe) # stored as a buffer, not a model parameter
    

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        # Just add the positional encodings to the input embeddings
        return x + self.pe[:, :seq_length]

class PositionalEmbedding(nn.Module):
    """
    Here we have a trainable matrix of shape (max_seq_len, embed_dim) that we learn during training.
    Each poistion 0,..., max_len_seq -1 return a learned embedding 
    This can potentially fir the data better (The model can learn whatever mapping from position 
    to embedding it wants), but it may not extrapolate as grafeully to positions beyong max_len_seq
    """
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(embedding_dim=embed_dim, num_embeddings=max_seq_len)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        positions = torch.arange(seq_length, device=to_device())
        positions = self.pe(positions) # shape (seq_length, embed_dim)
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
        # 1) Embedding look-up table
        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)

        # Initialize cls token parameter
        # Optional CLS token (common in BERT-like models)
        if self.pool == 'cls':
            #self.cls_token = ... # Institate an nn.Parameter with size: 1,1,embed_dim
            max_seq_len +=1
        

         # 3) Positional Encoding or Embedding
        if self.pos_enc == 'fixed':
            self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif self.pos_enc == 'learnable':
            self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        # 4) Stack of encoder blocks
        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        # 5) Final classifier
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length), integer token IDs

        # 1) Convert token IDs to embeddings
        tokens = self.token_embedding(x) # (batch_size, seq_length, embed_dim)
        batch_size, seq_length, embed_dim = tokens.size()

        # Include cls token in the input sequence
        # 2) If using a CLS token, prepend it
        ####################### insert code here #######################
        if self.pool == 'cls':
            # HINT: repeat the cls token of the batch dimension
            pass
        ################################################################


        # 3) Add positional information
        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        # 4) Pass through stacked encoder blocks
        x = self.transformer_blocks(x)

         # 5) Pooling across sequence
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
            # 6) Classifier output
        return self.classifier(x)
    


def test_attention():
    torch.manual_seed(0)  # for reproducibility

    # Define dimensions
    batch_size = 2
    seq_length = 5
    embed_dim = 16
    num_heads = 4
     # 1) Construct random input
    # Create a random input tensor: $$x \in \mathbb{R}^{B \times S \times E}$$
    x = torch.randn(batch_size, seq_length, embed_dim)
    # 2) Instantiate the Attention module
    # Initialize the Attention module
    attention_layer = Attention(embed_dim=embed_dim, num_heads=num_heads)
    # 3) Forward pass
    # Pass the input through the attention module
    output = attention_layer(x)

    # Print shapes to help you see what happens at each stage
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_attention()