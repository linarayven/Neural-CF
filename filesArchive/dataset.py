import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MovieDataset(Dataset):
    def __init__(self, ratings_path, movies_path, users_path=None):
        # Чтение файлов
        self.ratings = self._read_file(ratings_path, ['userId', 'movieId', 'rating', 'timestamp'])
        self.movies = self._read_file(movies_path, ['movieId', 'title', 'genres'])
        if users_path:
            self.users = self._read_file(users_path, ['userId', 'gender', 'age', 'occupation', 'zip'])
        else:
            self.users = None

        # Кодирование пользователей и фильмов
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.ratings['userId'] = self.user_encoder.fit_transform(self.ratings['userId'])
        self.ratings['movieId'] = self.movie_encoder.fit_transform(self.ratings['movieId'])

        # Сохраняем размеры
        self.num_users = len(self.user_encoder.classes_)
        self.num_movies = len(self.movie_encoder.classes_)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.ratings.iloc[idx]['userId']
        movie = self.ratings.iloc[idx]['movieId']
        rating = self.ratings.iloc[idx]['rating']
        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(movie, dtype=torch.long), \
               torch.tensor(rating, dtype=torch.float)

    def _read_file(self, path, col_names):
        ext = os.path.splitext(path)[1]
        if ext == '.dat':
            df = pd.read_csv(path, sep='::', engine='python', names=col_names, encoding='utf-8')
        elif ext == '.csv':
            df = pd.read_csv(path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return df

def get_dataloader(ratings_path, movies_path, users_path=None, batch_size=64):
    dataset = MovieDataset(ratings_path, movies_path, users_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.num_users, dataset.num_movies, dataset.user_encoder, dataset.movie_encoder, dataset
