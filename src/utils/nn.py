from typing import Union, Dict
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

import numpy as np


def load_word_vectors(path: Union[str, Path]) -> dict:
    word_vectors = {}
    with open(path, 'rt') as f:
        for line in f:
            word, *vector = line.split(' ')
            word_vectors[word] = np.array(vector)

    return word_vectors


def create_embedding_matrix(word_index: Dict[str, int], word_vectors: WordEmbeddingsKeyedVectors) -> np.ndarray:
    '''
    Populates a numpy 2d array with word vectors extracted from a Gensim model.
    '''
    embeddings = np.zeros((len(word_index) + 1, word_vectors.vector_size))

    for word, i in word_index.items():
        if word is not word_vectors:
            embeddings[i] = word_vectors[word]

    return embeddings


class ScoreScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, y):
        self._min = np.min(y)
        self._max = np.max(y)
        assert self._max != 0

        return self

    def transform(self, y):
        return y / self._max

    def inverse_transform(self, y):
        scores = np.round(y * self._max).astype(np.int).flatten()
        return np.clip(scores, self._min, self._max)


def predict_scores(X, model, scaler):
    y = model.predict_proba(X)
    scores = scaler.inverse_transform(y)

    return scores
