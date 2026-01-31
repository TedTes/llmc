import torch 
import torch.nn as nn
import torch.nn.functional as F 
from tourch.utils.data import Dataset , DataLoader
import math
import numpy as np 

# Model hyperparameters
vocab_size = 100        # Number of unique characters/tokens
embed_dim = 128         # Embedding dimension
num_heads = 4           # Number of attention heads
num_layers = 4          # Number of transformer blocks
ff_dim = 512            # Feed-forward hidden dimension (4x embed_dim)
max_seq_len = 128       # Maximum sequence length


# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch:i for i,ch in enumerate(self.chars)}
        self.idx_to_char = {i:ch for i, ch in enumerate(self.chars)}


    def encode(self,text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

    