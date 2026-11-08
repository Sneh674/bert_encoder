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

word_id = 25
vector = embedding(torch.tensor(word_id))
print(vector)
print(vector.shape) #64

word_ids = torch.tensor([25, 30, 45])
vectors = embedding(word_ids)
# print(vectors)
print(vectors.shape)  # [3, 64]

all_embeddings = embedding.weight
print(all_embeddings.shape)  # [56792, 64]

vector = embedding.weight[25]
print(vector)
print(vector.shape)  # [64]