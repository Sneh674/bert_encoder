import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

vocab_size = 30522
embedding_dim = 768

embedding_layer = EmbeddingLayer(
    vocab_size,
    embedding_dim
)

input_ids = torch.tensor([
    [101, 4937, 2003, 7777],
    [101, 2023, 2003, 1037]
])


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(seq_length)

        return self.embedding(positions)


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


# not needed - ignore this for now
# class TokenTypeEmbedding(nn.Module):

#     def __init__(self, embedding_dim):
#         super().__init__()

#         self.embedding = nn.Embedding(
#             2,
#             embedding_dim
#         )

#     def forward(self, token_type_ids):

#         return self.embedding(token_type_ids)

# token_type_ids = torch.tensor([
#     [0, 0, 0, 0, 1, 1, 1]
# ])

# segment_vectors = TokenTypeEmbedding(768)(token_type_ids)
# print({"segment_vectors": segment_vectors})
# print({"segment_vectors_shape": segment_vectors.shape})
    