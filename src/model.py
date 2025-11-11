import torch
import torch.nn as nn


class NeuralCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=32, hidden_sizes=None):
        super(NeuralCF, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # Embedding слои
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # MLP
        input_size = embedding_size * 2
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size

        # Финальный слой
        layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        x = torch.cat([user_emb, movie_emb], dim=-1)
        rating = self.mlp(x)

        # Масштабируем к диапазону [1,5]
        rating = torch.sigmoid(rating) * 4 + 1
        return rating.squeeze()
