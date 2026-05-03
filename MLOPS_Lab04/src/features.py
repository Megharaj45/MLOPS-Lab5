import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class RatingFeatures:

    def __init__(self):
        self.ratings_matrix = None
        self.similarity_matrix = None
        self.user_ids = None

    def fit(self, df):
        # Create pivot table
        self.ratings_matrix = df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )

        self.user_ids = self.ratings_matrix.index.values

        # Compute similarity
        self.similarity_matrix = cosine_similarity(self.ratings_matrix)

        # remove self similarity
        np.fill_diagonal(self.similarity_matrix, 0)

        print("Feature matrix created:", self.ratings_matrix.shape)

    def get_similar_users(self, user_id, n=5):
        idx = np.where(self.user_ids == user_id)[0]

        if len(idx) == 0:
            return []

        idx = idx[0]
        scores = self.similarity_matrix[idx]

        top = np.argsort(scores)[-n:][::-1]

        return [(int(self.user_ids[i]), float(scores[i])) for i in top]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print("Saved model:", path)

    @staticmethod
    def load(path):
        return joblib.load(path)