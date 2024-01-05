# Captures positional information of tokens in a sequence
import math
import torch
from pprint import pprint

# Positional encoding logic as described in the vanilla Transformer paper 
def generate_encoding(input_size, d_model, factor_n = 10000):
    encodings = []

    # Loop over all token sequences
    for L in range(0,input_size):
        encoding_buffer = []
        for i2 in range(0, d_model, 2):
            # Sine terms are always added
            encoding_buffer.append(math.sin(L/pow(factor_n,i2/d_model)))
            # When i2 == d_model-1, cosine term is only added if d_model is even
            if(i2 < d_model-1 or d_model%2 == 0):
                encoding_buffer.append(math.cos(L/pow(factor_n,i2/d_model)))
        encodings.append(encoding_buffer)
    return encodings

# Optimised version without loops
def generate_encoding_optimised(input_size, d_model, factor_n = 10000):
    # Generate all divisor and constants in tensors
    divisor_index = torch.arange(0, d_model, 2)
    divisors = torch.exp(divisor_index / -(d_model) * math.log(factor_n))
    constants = torch.arange(0, input_size).unsqueeze(1)
    
    encodings = torch.zeros(input_size, d_model)

    # Calculate encoding value at all even indices for each input position
    encodings[:, 0::2] = torch.sin(divisors*constants)
    # Calculate encoding value at all odd indices for each input position; handle dimension mismatch at edge cases
    if(d_model % 2 == 0):
        encodings[:, 1::2] = torch.cos(divisors*constants)
    else:
        encodings[:, 1::2] = torch.cos(divisors*constants)[:, :-1]

    return encodings

input_size = 10
d_model = 3
factor_n = 100


print('Straight forward encode:')
pprint(generate_encoding(input_size, d_model, factor_n))

print('========================\n'+\
      'Optimised:')
print(generate_encoding_optimised(input_size, d_model, factor_n))
