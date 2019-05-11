from typing import Tuple, Sequence, Dict, Callable, Any, List, Pattern, Union, Iterable, overload
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import partial, wraps
from pathlib import Path
import re

import nltk
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import toolz as tz
from pattern.en import parse
from pattern.web import plaintext
import toolz.curried as tzc
import bs4
import requests

QUOTES_PATTERN = [(r'\\\"|\\\'', '')]
PARENTHESIS_PATTERN = [(r'[]}{)(]', '')]


def get_raw_english_contractions():
    resp = requests.get('https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions')
    soup = bs4.BeautifulSoup(resp.text)
    table = soup.find('table', attrs={'class': 'wikitable'})
    table.find('tbody')
    data = []

    for row in table.find_all('tr'):
        cols = row.find_all('td')
        data.append([
            c.get_text()\
                .strip()\
                .replace(r'\[.+\]', '')

            for c in cols
        ])

    return data


def get_curated_english_contractions(path: Union[str, Path]):
    with open(path) as f:
        reader = csv.reader(f)
        contractions = [row for row in reader]

    return contractions


def remove_quotes(text, pattern: Pattern = QUOTES_PATTERN):
    '''
    removes escaped quotes. (\' or \")
    '''
    text = pattern.sub('', text)

    return text


def clean_html(text, linebreaks=1):
    '''
    removes html tags.
    '''

    return plaintext(text, linebreaks=linebreaks)


@tz.curry
def filter_stopwords(stopwords, tokens):
    return tz.filter(lambda t: t not in stopwords, tokens)


class RegexpReplacer(object):
    def __init__(self, patterns: Sequence[Tuple[Union[str, re.Pattern], str]]):
        self.patterns = [(re.compile(p), r) for p, r in patterns]

    def replace(self, text: str) -> str:
        for pattern, repl in self.patterns:
            text = pattern.sub(repl, text)

        return text

    def __call__(self, text):
        return self.replace(text)


# def process_text(text, func):
#     '''
#     clean and tokenize a text.
#     tokenize: bool indicate if the function should tokenize.
#     resolve: bool indicates if the iterator should be outputted to a list or not.
#     '''

#     return func(text)

class CorpusNormalizer(object):
    def __init__(self, *steps: Callable):
        self._steps = steps


    def transform(self,
                  corpus: Iterable[str],
                  map: Callable[[Callable, Iterable], Iterable] = tz.map,
                  collect=None) -> Iterable[str]:
        '''
        Process a corpus, represented as an iterable of text into a clean and tokenized corpus.
        Downstream tasks can be mapped to the return iterable.
        You can provide a custom map, for example to process the items in parallel.
        '''

        func = tz.compose(collect or tz.identity, *reversed(self._steps)) # compose applies last step first.
        # apply_steps = partial(process_text, func=func)
        processed_corpus = map(func, corpus)

        return processed_corpus


def create_corpus_processor(*steps):
    '''
    Produces a function that can be applied to a corpus of document, applying each step in series to each document.
    '''

    def process_corpus(corpus: Iterable[str],
                       map: Callable[[Callable, Iterable], Iterable] = tz.map,
                       collect=None) -> Iterable[str]:
        '''
        Process a corpus, represented as an iterable of text into a clean and tokenized corpus.
        Downstream tasks can be mapped to the return iterable.
        You can provide a custom map, for example to process the items in parallel.
        '''

        func = tz.compose(collect or tz.identity, *reversed(steps)) # compose applies last step first.
        # apply_steps = partial(process_text, func=func)
        processed_corpus = map(func, corpus)

        return processed_corpus

    return process_corpus


@tz.curry
def concatn(n, seqs):
    '''
    Concat nested sequences up to level n
    '''

    for _ in range(n):
        seqs = tz.concat(seqs)

    return seqs


def doc2bow(doc):
    '''
    Generates Bag of Word features
    '''

    return {word: True for word in doc}


def doc2boc(doc):
    '''
    Generates Bag of Counts features
    '''

    return Counter(doc)
