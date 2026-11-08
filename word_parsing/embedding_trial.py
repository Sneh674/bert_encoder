import torch
import torch.nn as nn
import json

with open("word_to_id.json", "r") as f:
    word_to_id = json.load(f)

with open("id_to_word.json", "r") as f:
    id_to_word = json.load(f)

with open("word_ids.json", "r") as f:
    word_ids = json.load(f)

print(len(word_to_id))

vocab_size = len(word_to_id)
embedding_dim = 64

embedding = nn.Embedding(vocab_size, embedding_dim)