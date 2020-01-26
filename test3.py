from sklearn.metrics import log_loss
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore");
from matplotlib import pyplot as plt

from handling import HandlingText

sample_submission = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_sample_submission.csv')
train_data = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_train.tsv', sep='\t')
test_data = pd.read_csv(r'C:/Users/Xiaomi/JN/mail.ru/products_sentiment_test.tsv', sep='\t')

train_data.rename(columns={train_data.columns[0]: "x", train_data.columns[1]: "y"}, inplace=True)
test_data.rename(columns={test_data.columns[0]: "id", test_data.columns[1]: "x"}, inplace=True)

print('train counter:', list(train_data['y']).count(1), 'of', len((train_data)))

train_data_x = []
for data in train_data['x']:
    train_data_x.append(HandlingText(data, ['clean', 'replace']))
test_data_x = []
for data in test_data['x']:
    test_data_x.append(HandlingText(data, ['clean', 'replace']))

from sklearn.feature_extraction.text import TfidfVectorizer

counter = TfidfVectorizer(ngram_range=(1,3))
train_counts = counter.fit_transform(train_data_x + test_data_x)
print('features:', counter.get_feature_names())
print('shape:', train_counts.shape)
print('voc:', train_counts.toarray())
print('freq:', train_counts.toarray().sum(axis=0))

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectFromModel

model = LogisticRegression(penalty='l2', C=10.0)
model.fit(train_counts.toarray()[:len(train_data_x)], train_data['y'])
X_train_word_average1 = SelectFromModel(model, prefit=True).transform(train_counts.toarray()[:len(train_data_x)])
X_test_word_average1 = SelectFromModel(model, prefit=True).transform(train_counts.toarray()[len(train_data_x):])
print('shape2:', X_train_word_average1.shape)


C_list = [0.01 + i * 0.01 for i in range(1000)]
C_list = [10.0]
print('logspace:', C_list)
grid = {"C": C_list, "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=4, scoring='neg_log_loss')
logreg_cv.fit(X_train_word_average1, train_data['y'])
print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
print("accuracy :", logreg_cv.best_score_)

model = LogisticRegression(penalty='l2', C=10.0)
model.fit(X_train_word_average1, train_data['y'])
pred = model.predict_proba(X_test_word_average1)[:, 1]
submission = []
print(len(pred))
for index in range(len(pred)):
    submission.append([test_data['id'][index], pred[index]])
dt_submission = pd.DataFrame(submission, columns=['Id', 'y'])
dt_submission.to_csv(r'C:/Users/Xiaomi/JN/mail.ru/my_submission_1.csv', index=False)
print(dt_submission)

from handling import w2v_tokenize_text