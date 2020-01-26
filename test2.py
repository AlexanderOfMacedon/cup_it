from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import nltk
import warnings

warnings.filterwarnings("ignore");

from handling import HandlingText
from gensim.matutils import unitvec
from gensim.models import Word2Vec, KeyedVectors


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


def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


sample_submission = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_sample_submission.csv')
train_data = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_train.tsv', sep='\t')
test_data = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_test.tsv', sep='\t')

train_data.rename(columns={train_data.columns[0]: "x", train_data.columns[1]: "y"}, inplace=True)
test_data.rename(columns={test_data.columns[0]: "id", test_data.columns[1]: "x"}, inplace=True)

train_data_x = []
for data in train_data['x']:
    train_data_x.append(w2v_tokenize_text(HandlingText(data, ['clean', 'replace'])))
test_data_x = []
for data in test_data['x']:
    test_data_x.append(w2v_tokenize_text(HandlingText(data, ['clean', 'replace'])))
print(train_data_x)

wv = KeyedVectors.load_word2vec_format("C:/Users/Xiaomi/mail.ru/GoogleNews-vectors-negative300.bin.gz", binary=True)

X_train_word_average = word_averaging_list(wv, train_data_x)
X_test_word_average = word_averaging_list(wv, test_data_x)

print('shape1:', X_train_word_average.shape)

from sklearn.model_selection import GridSearchCV
C_list = [0.01, 0.1, 1.0, 10.0, 100.0]
grid = {"C": C_list, "penalty": ["l1", "l2"]}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=4, scoring='neg_log_loss')
logreg_cv.fit(X_train_word_average, train_data['y'])
print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
print("accuracy :", logreg_cv.best_score_)

# from sklearn.feature_selection import SelectFromModel
#
# model = LogisticRegression(penalty='l2', C=1.0)
# model.fit(X_train_word_average, train_data['y'])
# X_train_word_average1 = SelectFromModel(model, prefit=True).transform(X_train_word_average)
# X_test_word_average1 = SelectFromModel(model, prefit=True).transform(X_test_word_average)
# print('shape2:', X_train_word_average1.shape)

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=4, scoring='neg_log_loss')
logreg_cv.fit(X_train_word_average, train_data['y'])
print("tuned hpyerparameters1 :(best parameters) ", logreg_cv.best_params_)
print("accuracy1 :", logreg_cv.best_score_)

model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X_train_word_average, train_data['y'])

pred = model.predict(X_test_word_average)
submission = []
for index in range(len(pred)):
    submission.append([test_data['id'][index], pred[index]])
dt_submission = pd.DataFrame(submission, columns=['Id', 'y'])
dt_submission.to_csv(r'C:/Users/Xiaomi/JN/mail.ru/my_submission_1.csv', index=False)
print(dt_submission)
