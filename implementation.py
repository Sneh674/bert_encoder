import torch
import torch.nn as nn
from bert_embedding import BertEmbeddings
import json
from read_txt import read_txt

with open("word_parsing/word_to_id.json", "r") as f:
    word_to_id = json.load(f)

with open("word_parsing/id_to_word.json", "r") as f:
    id_to_word = json.load(f)

with open("word_parsing/word_ids.json", "r") as f:
    word_ids = json.load(f)

print(len(word_to_id))

vocab_size = len(word_to_id)
embedding_dim = 64
max_seq_length = 10
dropout = 0.0

training_tokens=read_txt("word_parsing/training.txt",max_seq_length)
embedding=BertEmbeddings(vocab_size,embedding_dim,max_seq_length,dropout)

epochs = 10

# print(word_to_id["[CLS]"])
# print(word_to_id["the"])
# print(word_to_id["[UNK]"])
# if "seswwwsx" in word_to_id:
#     print(word_to_id["seswwwsx"])
# else:
#     print("seswwwsx not found in word_to_id")
word_id_sequences = []
for sequence in training_tokens:
    word_id_sequences.append([word_to_id.get(word, word_to_id["[UNK]"]) for word in sequence])

# print(word_id_sequences)


# Convert to PyTorch tensor
input_ids = torch.tensor(
    word_id_sequences,
    dtype=torch.long
)

print("Input IDs shape:", input_ids.shape)

# Initialize BERT embeddings
embedding = BertEmbeddings(
    vocab_size,
    embedding_dim,
    max_seq_length,
    dropout
)

# Generate embeddings
# output = embedding(input_ids)
output = []
for i in range(len(input_ids)):
    print("Processing sequence", i)
    # print("input_ids[i] shape:", input_ids[i].size(0))
    # embedding(input_ids[i])
    output.append(embedding(input_ids[i]))


print("\nEmbedding output:")
print(output)

print("\nOutput shape:", len(output))
print("\nSingle output shape:", output[0].shape)
# print("\nOutput shape:", [item.shape for item in output])