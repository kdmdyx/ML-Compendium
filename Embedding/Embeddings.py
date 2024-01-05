import torch

from torch import Tensor

def generate_vocabs(input: str):

    # Remove punctuations
    buffer = []
    to_filter = {',', '.', ';', ':', "\"", '!'}
    for cha in input:
        if cha not in to_filter:
            buffer.append(cha)

    filter_out = ''.join(buffer)
    
    # Force lower case letters, split tokens, and sort list alphabetically
    token_list = [token.lower() for token in filter_out.split(' ')]
    token_list.sort()

    # Establish vocab dictionary via enmueration
    return {token: index for token, index in enumerate(token_list)}

# Basic embedding, converting tokens of defined language to n-dimensional vectors
class Embedding(torch.nn.Module):
    def __init__(self,
                 rs:str,
                 d_model:int):
        
        super.__init__()
        self.d_model = d_model
        self.vocabs = generate_vocabs(rs)
        self.lookup = torch.rand(len(self.vocabs.keys), d_model)

    # Map string tokens to integer indices in dictionary
    def tokens2index(self, tokens):
        return [self.vocabs[token] for token in tokens]
    
    # Lookup embeded vectors for sequence of indices in current embedding
    def forward(self, sequence: Tensor):
        return self.lookup[sequence]


#----------------- Test Zone -------------------
print(generate_vocabs("Apple, Banana and Dad's Cadilac!"))