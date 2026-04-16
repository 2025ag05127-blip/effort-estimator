#!/usr/bin/env python3
"""
Fine-tune Jarvis on training_data.txt collected from DeepSeek.
This version automatically handles very small datasets.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# ========== CONFIG ==========
TRAINING_DATA_FILE = "training_data.txt"
OUTPUT_MODEL_FILE = "jarvis_model_finetuned.pth"
TOKENIZER_FILE = "tokenizer.pkl"
BLOCK_SIZE = 64          # smaller default for faster training
BATCH_SIZE = 4
MAX_ITERS = 1000         # enough for small dataset
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== MODEL ==========
class SimpleJarvisGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._make_block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _make_block(self, n_embd, n_head):
        return nn.TransformerDecoderLayer(n_embd, n_head, batch_first=True, dropout=0.1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos[:, :T])
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        x = self.blocks(x, tgt_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.7):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ========== LOAD DATA ==========
print("Loading training data...")
with open(TRAINING_DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read().strip()

if not text:
    print("Error: training_data.txt is empty. Run jarvis_agent.py in LEARNING_MODE first.")
    exit(1)

# Build tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

# Save tokenizer
with open(TOKENIZER_FILE, 'wb') as f:
    pickle.dump((stoi, itos), f)
print(f"Vocabulary size: {vocab_size} | Total chars: {len(text)}")

# Encode
data = torch.tensor(encode(text), dtype=torch.long)
data_len = len(data)

# Adjust block size if data is too small
if data_len < BLOCK_SIZE + 1:
    BLOCK_SIZE = max(1, data_len - 2)
    print(f"Data small → block_size reduced to {BLOCK_SIZE}")
    if BLOCK_SIZE < 2:
        print("ERROR: Need at least 3 characters in training file.")
        exit(1)

# Split (use all for training if tiny)
if data_len < 100:
    train_data = data
    val_data = data
else:
    n = int(0.9 * data_len)
    train_data = data[:n]
    val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_start = len(data) - BLOCK_SIZE - 1
    if max_start <= 0:
        ix = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    else:
        ix = torch.randint(0, max_start, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ========== TRAIN ==========
model = SimpleJarvisGPT(vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training on {DEVICE} for {MAX_ITERS} iters (block_size={BLOCK_SIZE})...")
for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            _, loss = model(xb, yb)
            print(f"Step {step}: val loss = {loss.item():.4f}")
        model.train()
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
print(f"\n✅ Fine-tuned model saved to {OUTPUT_MODEL_FILE}")
print(f"✅ Tokenizer saved to {TOKENIZER_FILE}")

# Quick test
model.eval()
test_prompt = "user: what time is it?\njarvis:"
ctx = torch.tensor([stoi.get(ch, 0) for ch in test_prompt], dtype=torch.long).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=40)
response = ''.join([itos[int(i)] for i in out[0].tolist()])
if "jarvis:" in response:
    response = response.split("jarvis:")[-1].strip()
print(f"Test response: {response}")