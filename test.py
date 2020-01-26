from sklearn.metrics import log_loss
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

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
    train_data_x.append(HandlingText(data, ['clean', 'replace', 'stem']))
test_data_x = []
for data in test_data['x']:
    test_data_x.append(HandlingText(data, ['clean', 'replace', 'stem']))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# counter
counter = TfidfVectorizer()
train_counts = counter.fit_transform(train_data_x + test_data_x)
print('features:', counter.get_feature_names())
print('shape:', train_counts.shape)
print('voc:', train_counts.toarray())
print('freq:', train_counts.toarray().sum(axis=0))

from sklearn.model_selection import GridSearchCV

grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 lasso l2 ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=4, scoring='neg_log_loss')
logreg_cv.fit(train_counts.toarray()[:len(train_data_x)], train_data['y'])
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

# test_counts = counter.fit_transform(test_data_x)

model = LogisticRegression(penalty='l2', C=10.0)
model.fit(train_counts.toarray()[:len(train_data_x)], train_data['y'])
pred = model.predict_proba(train_counts.toarray()[len(train_data_x):])[:,1]
submission = []
print(len(pred))
for index in range(len(pred)):
    submission.append([test_data['id'][index], pred[index]])
dt_submission = pd.DataFrame(submission, columns = ['Id', 'y'])
dt_submission.to_csv(r'C:/Users/Xiaomi/JN/mail.ru/my_submission_1.csv', index=False)
print(dt_submission)


# model
# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LogisticRegression(penalty='l2', C=2.0))])
#
# text_clf.fit(train_data_x, train_data['y'])
#
# score0 = train_data['y']
# score1 = text_clf.predict_proba(train_data_x)[:,1]
#
#
#
#
# score2 = text_clf.predict_proba(test_data_x)[:,1]

# submission = []
# for index in range(len(score2)):
#     submission.append([test_data['id'][index], score2[index]])
# dt_submission = pd.DataFrame(submission, columns = ['Id', 'y'])
# dt_submission.to_csv(r'C:/Users/Xiaomi/JN/mail.ru/my_submission_1.csv', index=False)
# print(dt_submission)
from sklearn.model_selection import cross_val_score, KFold

# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LogisticRegression(penalty='l2', C=4))])
# kf = KFold(n_splits=4)
# results = cross_val_score(text_clf, train_data_x, train_data['y'], cv=kf, scoring='neg_log_loss')
# print(np.mean(-results))
