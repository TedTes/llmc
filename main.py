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



# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    
    def forward(self, x):
        batch_size, seq_len , embed_dim = x.shape

        # compute Q,K,V

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads , self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch, num_heads, seq_len, head_dim)