from nltk.corpus import stopwords
from string import punctuation

symbols_replace = [',', '-', '!', '_', '"', '.', ';', '—', '(', ')', '[', ']', '«', '»', '&', '$', '{', '}', '?', ':',
                   '*', '/', '+', '-', '|', '?', '>', '<', '#', '№', '@', '%', '\'', '\\']
symbols_replace += [str(x) for x in range(10)]
stop_words = stopwords.words('english') + list(punctuation)

