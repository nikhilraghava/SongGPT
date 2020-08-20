import math
import torch
import logging
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import sample
from mingpt.utils import set_seed
from torch.utils.data import Dataset
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

# Set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# Make it deterministic
set_seed(42)

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # We're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# Spatial extent of the model for its context
block_size = 128

text = open('input.txt', 'r').read()
train_dataset = CharDataset(text, block_size)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

# Initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=200, batch_size=512, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset)*block_size,
                      num_workers=0)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()

context = "Ooh ya"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)