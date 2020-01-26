from nltk.stem.snowball import SnowballStemmer

import my_stopwords

def CleanStopWordsInText(text):
    splitting_text_list = text.split(' ')
    return ' '.join(x for x in splitting_text_list if x not in my_stopwords.stop_words)

def ReplaceSymbolsInText(text):
    text_ret = text
    for symbol in my_stopwords.symbols_replace:
        text_ret = text_ret.replace(symbol, ' ')
    return text_ret

def StemText(text):
    text_word_list = text.split()
    stemmer = SnowballStemmer('english')
    for index in range(len(text_word_list)):
        text_word_list[index] = stemmer.stem(text_word_list[index])
    return ' '.join(text_word_list)

def HandlingText(text, options):
    temp_text = text
    for option in options:
        if option == 'clean':
            temp_text = CleanStopWordsInText(temp_text)
        if option == 'replace':
            temp_text = ReplaceSymbolsInText(temp_text)
        if option == 'stem':
            temp_text = StemText(temp_text)
    return temp_text

from gensim.matutils import unitvec
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

def word_averaging(wv, words):
    all_words, mean = set(), []

    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size, )

    mean = unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean


def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list])

import nltk

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens