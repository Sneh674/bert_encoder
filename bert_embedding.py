import torch
import torch.nn as nn
from embedding import EmbeddingLayer, PositionalEmbedding


class BertEmbeddings(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_seq_length, dropout_prob=0.1):
        super().__init__()

        # Token embeddings
        self.token_embedding = EmbeddingLayer(
            vocab_size,
            embedding_dim
        )

        # Positional embeddings
        self.position_embedding = PositionalEmbedding(
            max_seq_length,
            embedding_dim
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        # print("input ids shape:",input_ids.shape)

        # batch_size, seq_length = input_ids.shape
        # seq_length = input_ids.shape
        seq_length = input_ids.size(0)
        # print("num_sequences:", num_sequences)
        print("seq_length:", seq_length)

        # Generate position IDs
        positions = torch.arange(
            seq_length,
            device=input_ids.device
        )

        # Get token embeddings
        token_vectors = self.token_embedding(input_ids)

        # Get positional embeddings
        position_vectors = self.position_embedding(positions)

        # Combine token and positional embeddings
        embeddings = token_vectors + position_vectors

        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings


# token_embedding = EmbeddingLayer(30522, 768)

# position_embedding = PositionalEmbedding(512, 768)

# input_ids = torch.tensor([
#     [101, 4937, 2003, 7777],
#     [101, 2023, 2003, 1037]
# ])

# token_vectors = token_embedding(input_ids)
# print({"token_vectors": token_vectors})
# print({"token_vectors_shape": token_vectors.shape})

# position_vectors = position_embedding(input_ids)
# print({"position_vectors": position_vectors})
# print({"position_vectors_shape": position_vectors.shape})


# final_embeddings = token_vectors + position_vectors

# print({"final_embeddings": final_embeddings})
# print({"final_embeddings_shape": final_embeddings.shape})