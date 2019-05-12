from typing import Union, Dict, Iterable
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
from gensim.models import FastText
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import History
import numpy as np

from .corpus import Corpus


class ModelWrapper:
    '''
    Wraps a keras model and expose a common interface.
    '''

    model: Union[Sequential, Model]
    name: str

    def __init__(self, tokenizers: dict, use_softmax: bool = True, class_weight: dict = None):
        self.tokenizers = tokenizers
        self._score_scaler = ScoreScaler()
        self.use_softmax = use_softmax
        self.class_weight = class_weight

    def compile(self) -> Union[Sequential, Model]:
        '''
        Compiles the model
        '''
        self.model = self._compile()

        return self.model

    def _compile(self) -> Union[Sequential, Model]:
        '''
        Implement to create the underlying Keras model.
        '''
        raise NotImplementedError

    def fit(self, x, y, *args, **kwargs) -> History:
        '''
        Calls the fit method of the keras model and returns the history.
        '''

        if self.use_softmax:
            y = y - 1
            class_weight = self.class_weight
        else:
            y = self._score_scaler.fit_transform(y)
            class_weight = None

        return self.model.fit(x, y, class_weight=class_weight, *args, **kwargs)

    def predict(self, x) -> np.ndarray:
        if self.use_softmax:
            cat_pred = self.model.predict_classes(x)
            y_pred = cat_pred + 1
        else:
            y_pred = predict_scores(x, self.model, self._score_scaler)

        return y_pred

    def text_to_padded_sequences(self, corpus: Corpus) -> Iterable:
        '''
        Extract the relevant information for the model from the corpus
        '''
        raise NotImplementedError


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


def train_fasttext(corpus_file: str, size: int, total_words: int, epochs: int, window=5) -> FastText:
    '''
    Train the Facebook FastText model on the sentences in the corpus file.
    '''
    vector_model = FastText(size=size, window=window, min_count=1, sg=1)
    vector_model.build_vocab(corpus_file=corpus_file)
    vector_model.train(corpus_file=corpus_file,
                       total_examples=vector_model.corpus_count,
                       total_words=total_words,
                       epochs=epochs)

    return vector_model
