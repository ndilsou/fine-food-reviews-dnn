from __future__ import annotations
from typing import Sequence, Tuple, Optional, Union, List, Iterator
import zipfile
from pathlib import Path
import os

import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from spacy.language import Language
import numpy as np
import nltk

class Corpus:
    '''
    Container for a corpus of Spacy Docs
    '''
    def __init__(self, nlp: Language, *, indexed_documents: Optional[Sequence[Tuple[Doc, int]]] = None,
                 documents_series: Optional[pd.Series] = None):
        self.nlp = nlp
        self._documents: pd.Series

        if indexed_documents is not None:
            docs, index = zip(*indexed_documents)
            self._documents = pd.Series(docs, index)
        elif documents_series is not None:
            self._documents = documents_series
        else:
            raise ValueError('set one of indexed_documents or documents_series')

    @property
    def index(self) -> pd.Index:
        return self._documents.index

    def __iter__(self) -> Iterator[Doc]:
        for doc in self._documents:
            yield doc

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return repr(self._documents)

    def __getitem__(self, index: Union[Sequence[int], int]) -> Union[Doc, Corpus]:
        if isinstance(index, int):
            result = self._documents[index]
        else:
            result = Corpus(self.nlp, documents_series=self._select(index))

        return result

    def _select(self, index: Sequence[int] = None) -> pd.Series:
        document: pd.Series

        if index:
            documents = self._documents[index]
        else:
            documents = self._documents

        return documents

    def sentences_generator(self, index: Optional[Sequence[int]] = None) -> Iterator[str]:
        for doc in self._select(index):
            for sent in doc.sents:
                yield sent.text

    def sentences(self, index: Optional[Sequence[int]] = None) -> Sequence[str]:
        return list(self.sentences_generator(index))

    def texts_generator(self, index: Optional[Sequence[int]] = None) -> Iterator[str]:
        for doc in self._select(index):
            yield doc.text

    def texts(self, index: Optional[Sequence[int]] = None) -> Sequence[str]:
        return list(self.texts_generator(index))

    def pos_tags_generator(self, index: Optional[Sequence[int]] = None) -> Iterator[str]:
        for doc in self._select(index):
            yield [token.tag_ for token in doc]

    def pos_tags(self, index: Optional[Sequence[int]] = None) -> Sequence[str]:
        return list(self.pos_tags_generator(index))

    def count_sentences(self) -> int:
        return np.sum(len(doc.sents) for doc in self._documents)

    def count_words(self) -> int:
        return np.sum(len(doc) for doc in self._documents)

    def save(self, path: Union[str, Path]) -> None:
        '''
        Archive the serialized document in a zip file.
        '''
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for i, doc in self._documents.iteritems():
                data = doc.to_bytes(exclude=['tensor'])
                archive.writestr(f'{i}.pickle', data)

    @classmethod
    def load(cls, nlp: Language, path: Union[str, Path]) -> Corpus:
        index = []
        docs = []
        with zipfile.ZipFile(path, 'r', compression=zipfile.ZIP_DEFLATED) as archive:
            for fname in archive.namelist():
                i, _ = os.path.splitext(fname)
                index.append(i)
                with archive.open(fname) as f:
                    doc = Doc(nlp.vocab).from_bytes(f.read())
                    docs.append(doc)
        indexed_documents = zip(docs, index)

        return cls(nlp, indexed_documents=indexed_documents)
