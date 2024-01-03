import torch
import math
import numpy as np

from Torch import Tensor
from Util import buildVocab

# Basic embedding, converting tokens of defined language to n-dimensional vectors
class Embedding(torch.nn.module):
    def __init__(self,
                 rs:str,
                 d_model:int):
        
        super.__init__()
        self.d_model = d_model
        self.vocabs = buildVocab(rs)
        self.lookup = torch.rand(len(self.vocabs.keys), d_model)

    # Map string tokens to integer indices in dictionary
    def tokens2index(self, tokens):
        return [self.vocabs[token] for token in tokens]
    
    # Lookup embeded vectors for sequence of indices in current embedding
    def forward(self, sequence: Tensor):
        return self.lookup[sequence]


# Embedding layer as implemented in the vanilla transformer paper 
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

#----------------- Test Zone -------------------
