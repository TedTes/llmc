import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset , DataLoader
import math
import numpy as np 

# Model hyperparameters
vocab_size = 100        # Number of unique characters/tokens
embed_dim = 128         # Embedding dimension
num_heads = 4           # Number of attention heads
num_layers = 4          # Number of transformer blocks
ff_dim = 512            # Feed-forward hidden dimension (4x embed_dim)
max_seq_len = 128       # Maximum sequence length



# Training hyperparameters
batch_size = 32
learning_rate = 3e-4
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Simple character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
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

        Q, K , V =  qkv[0] , qkv[1] , qkv[2];

        # Scaled dot-product attention
        scores =  (Q @  K.transpose(-2, -1) ) / math.sqrt(self.head_dim)

        # Causal mask (prevent attending to future tokens)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))


        # Apply softmax and compute weighted values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        return output
    

    # Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class SimpleLLM(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_heads , num_layers , ff_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, ff_dim)
        for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    
    def forward(self, idx):
        batch_size, seq_len  = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, seq_len, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        
        x = tok_emb + pos_emb

        # Pass through transformer blocks

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits



# Dataset for training
class TextDataset(Dataset):
    def __init__(self, text , tokenizer , max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = tokenizer.encode(text)
        
    
    def __len__(self):
        return len(self.data) - self.max_seq_len
    

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.max_seq_len+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y 



def generate_text(model, tokenizer, start_text , max_new_tokens=100, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():


        for _ in range(max_new_tokens):
            # Get predictions (use only last max_seq_len tokens)
            idx_cond = tokens[:, -max_seq_len:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim = 1)

    generated_tokens = tokens[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text     

text =  """Hello world! This is a simple example text for training our LLM. """ * 20
tokenizer = CharTokenizer(text)

vocab_size = tokenizer.vocab_size

# Initialize model
model = SimpleLLM(vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len).to(device)


dataset = TextDataset(text, tokenizer, max_seq_len)

dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)

#Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


# Demo: Generate text before and after training
print("\n--- Before Training ---")
print(generate_text(model, tokenizer, "Hello", max_new_tokens=50, temperature=0.8))

# Training loop
print(f"Training on {device}...")
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")






print("\n--- Training Complete ---")
print("\n--- After Training ---")
print(generate_text(model, tokenizer, "Hello", max_new_tokens=50, temperature=0.8))

# Test 1: Different starting point
print(generate_text(model, tokenizer, "This", max_new_tokens=50))

# Test 2: Partial word
print(generate_text(model, tokenizer, "exam", max_new_tokens=50))

# Test 3: Single letter
print(generate_text(model, tokenizer, "T", max_new_tokens=50))