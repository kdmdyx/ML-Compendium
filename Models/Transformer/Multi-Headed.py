import torch
import math

from torch import Tensor

from Misc.Encoding import Positional

# Embeddings 
class Embedding_Transformer(torch.nn.Module):
    def __init__(self,
                 vocab_size: int,               # Total amount of unique tokens to embed 
                 d_model: int):                 # Dimension of output vectors
        
        super.__init__()
        self.d_model = d_model
        self.lookup = torch.nn.Embedding(vocab_size, d_model)

    # Vanilla Transformer multiplies embedded token values by a scalar of the squareroot of d_model
    def forward(self, sequence: Tensor):

        return self.lookup[sequence] * math.sqrt(self.d_model) 

# Positional Encoding
class PostionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, dropout:float = 0.1, max_length:int = 5000):
        super.__init__()
        self.Dropout = torch.nn.Dropout(p=dropout)

        # Generate and stash the positional encoding map (n is hard-coded as 10k here *shrug*)
        pe = Positional.generate_encoding_optimised(max_length, d_model, 10000)
        pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x:Tensor):
        # Add to embedding and dropout
        x = x+self.pe[:, :(x.size(1))].requires_grad_(False)
        return self.dropout(x)





