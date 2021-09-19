#!/usr/bin/env python
# coding: utf-8

# __Use PYTHON’s Multinomial Naïve Bayes__ algorithm to build models to classify the customer reviews by   
# (1) sentiment (positive or negative)   
# (2) authenticity (true or fake, lie detection)  
# 
# For each of the two classification tasks, use MNB to build the models, and evaluate them using 10-fold cross validation methods. (5-fold is fine too) 
# 
# __Use CountVectorizer and Python. Create labeled data. Train the NB model and test it.__ 
# 
# __As part of your Results section__: For each model (lie detection and sentiment classification), report the 20 most indicative words that the models have learned. 
# 
# __As part of your Results (the techy part) and Conclusions (the non-techy part) include discussion of__: Based on these words, do you think the models have learned the concepts (lie or sentiment) that they are expected to learn?
# 
# __As Part of Results__: Also, compare the difficulty level of sentiment classification vs. lie detection. Discuss whether you believe computers can detect fake reviews by the words.

import pandas as pd
import re
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)

reviews = pd.read_csv('deception_data_converted_final.tsv', sep='\t')

reviews.shape

reviews.sample(5, random_state=6)

# Summary:
#    * 92 instances
#    * labels are located at fisrt two columns, one is for `lie`; the other is for `sentiment`
#    * back slash and apostrophe have to be removed

def clean_str(string):
    """
    expect: a string with leading and trailing apostrophe and a back slash within the string
    ex:"'I found worm \in one of the dishes.'"
    modify: Remove the trailing and leading apostrophe and back slash
    return: return a string
    """
    temp = re.sub(r'\\', '', string)
    result = temp.strip("'")
    return result


sentiment_mapper = {'n':0,'p':1}
lie_mapper = {'t': 0, 'f': 1}   # true = 0 ; fake = 1
reviews['review'] = reviews.review.map(clean_str)
reviews['lie_num'] = reviews.lie.map(lie_mapper)
reviews['sentiment_num'] = reviews.sentiment.map(sentiment_mapper)

reviews.sample(5, random_state=6)

reviews.sentiment.value_counts().plot(kind='bar', figsize=(8,6), fontsize=11)
plt.title('Raw Dataset', size=12)
plt.xlabel('Sentiment')
plt.ylabel('Counts')
# add annotation on each bar
for i in range(2):
    plt.text(x = i - 0.03 , y = reviews.sentiment.value_counts().values[i] + 0.4, 
             s = reviews.sentiment.value_counts().values[i])
plt.show()

reviews.lie.value_counts().plot(kind='bar', figsize=(8,6), fontsize=11)
plt.title('Raw Dataset', size=12)
plt.xlabel('lie')
plt.ylabel('Counts')
# add annotation on each bar
for i in range(2):
    plt.text(x = i - 0.03 , y = reviews.lie.value_counts().values[i] + 0.4, 
             s = reviews.lie.value_counts().values[i])
plt.show()



# ## Sentiment

# define X and y
X = reviews.review
y = reviews.sentiment_num


# check the shape
print(X.shape)
print(y.shape)


# ### Holdout set technique: Train/test split

from sklearn.model_selection import train_test_split # test_size=0.25 in default
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,)


# #### Features Contain Stop Words

# import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect


get_ipython().run_cell_magic('time', '', "# create a document-term matrix from all of the training data\nX_train_dtm = vect.fit_transform(X_train)\nprint('X_train_dtm shape: {}'.format(X_train_dtm.shape))\nX_test_dtm = vect.transform(X_test)\nprint('X_test_dtm shape: {}'.format(X_test_dtm.shape))")

print('There are {} tokens'.format(len(vect.get_feature_names())))
print(vect.get_feature_names()[:50])

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# Train the model and make a prediction\nnb.fit(X_train_dtm.toarray(), y_train)\ny_pred_class = nb.predict(X_test_dtm.toarray())')

from sklearn.metrics import accuracy_score
print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_pred_class)*100, 2)))


# Confusion Matrix
from sklearn.metrics import plot_confusion_matrix
maxtix = plot_confusion_matrix(nb, X_test_dtm.toarray(), y_test, cmap='Blues', values_format='.3g')


# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class, digits=4))


# #### Features Not Contain Stop Words

# import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_stop = CountVectorizer(stop_words='english')
vect_stop

get_ipython().run_cell_magic('time', '', "# create a document-term matrix from all of the training data\nX_train_dtm = vect_stop.fit_transform(X_train)\nprint('X_train_dtm shape: {}'.format(X_train_dtm.shape))\nX_test_dtm = vect_stop.transform(X_test)\nprint('X_test_dtm shape: {}'.format(X_test_dtm.shape))")

print('There are {} tokens'.format(len(vect_stop.get_feature_names())))
print(vect.get_feature_names()[:50])


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# Train the model and make a prediction\nnb.fit(X_train_dtm.toarray(), y_train)\ny_pred_class = nb.predict(X_test_dtm.toarray())')


from sklearn.metrics import accuracy_score
print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_pred_class)*100, 2)))


# Confusion Matrix
from sklearn.metrics import plot_confusion_matrix
maxtix = plot_confusion_matrix(nb, X_test_dtm.toarray(), y_test, cmap='Blues', values_format='.3g')

reviews.loc[y_test.index, 'sentiment']

y_test

y_test.value_counts()

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class, digits=4))

result = (y_test.value_counts()[0]/y_test.shape[0])
print('Base model(null model) for sentiment prediction in testing set: {}%'.format(round(result*100, 2)))


# #### K-fold Cross-validation

import numpy as np
np.random.seed(1)
print('Base model(null model) for sentiment prediction: {}%'.format((y.value_counts()/y.shape[0])[0]))


# 5 folds
get_ipython().run_cell_magic('time', '', "nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words='english')\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=5, scoring='accuracy').mean()\nprint('Accuracy: {}%'.format(round(result*100, 2)))")

get_ipython().run_cell_magic('time', '', "nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words='english')\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=5, scoring='recall').mean()\nprint('Recall: {}%'.format(round(result*100, 2)))")


# 10 folds
get_ipython().run_cell_magic('time', '', "nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words='english')\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=10, scoring='accuracy').mean()\nprint('Accuracy: {}%'.format(round(result*100, 2)))")

get_ipython().run_cell_magic('time', '', "nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words='english')\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=10, scoring='recall').mean()\nprint('Recall: {}%'.format(round(result*100, 2)))")


# #### Feature Ranking in MultinomialNB
X_tokens = vect_cv.get_feature_names()

# examine the first 50 tokens
print(X_tokens[0:50])

# examine the last 50 tokens
print(X_tokens[-50:])

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# train the model using X_dtm\nnb.fit(X_dtm, y)')

nb.feature_count_
# rows represent classes, columns represent tokens
print("Shape: {}".format(nb.feature_count_.shape))

# number of times each token appears across all 'neg' class
neg_token_count = nb.feature_count_[0, :]
neg_token_count

# number of times each token appears across all 'pos' class
pos_token_count = nb.feature_count_[1, :]
pos_token_count

# create a DataFrame of tokens with their separate atheism, forsale and med counts
tokens = pd.DataFrame({'token':X_tokens, 'neg':neg_token_count, 
                       'pos':pos_token_count}).set_index('token')


# examine 5 random DataFrame rows
tokens.sample(5, random_state=6)

# Naive Bayes counts the number of observations in each class
nb.class_count_

# Naive Bayes counts the number of observations for all classes
Total_vocab = tokens.shape[0]
print('Number of vocabulary learned: {}'.format(Total_vocab))

# calculate the condition probabilities using Laplace smoother (N + |vocab|)
# https://scikit-learn.org/stable/modules/naive_bayes.html
tokens['neg'] = (tokens.neg +1) / (nb.class_count_[0]+Total_vocab)
tokens['pos'] = (tokens.pos +1) / (nb.class_count_[1]+Total_vocab)
tokens['pos_ratio'] = tokens.pos/tokens.neg
tokens['neg_ratio'] = tokens.neg/tokens.pos
tokens = tokens.sort_values('pos_ratio', ascending=False)
tokens.head(20)

print('most indicative words for negative sentiment')
tokens.loc[:,'neg'].sort_values(ascending=False)[:20]

print('most indicative words for positive sentiment')
tokens.loc[:,'pos'].sort_values(ascending=False)[:20]



# ## Lie

# define X and y
X = reviews.review
y = reviews.lie_num

print(X.shape)
print(y.shape)


# ### Holdout set technique: Train/test split

from sklearn.model_selection import train_test_split # test_size=0.25 in default
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,)


# #### Features Contain Stop Words

# import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect

get_ipython().run_cell_magic('time', '', "# create a document-term matrix from all of the training data\nX_train_dtm = vect.fit_transform(X_train)\nprint('X_train_dtm shape: {}'.format(X_train_dtm.shape))\nX_test_dtm = vect.transform(X_test)\nprint('X_test_dtm shape: {}'.format(X_test_dtm.shape))")

print('There are {} tokens'.format(len(vect.get_feature_names())))
print(vect.get_feature_names()[:50])

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# Train the model and make a prediction\nnb.fit(X_train_dtm.toarray(), y_train)\ny_pred_class = nb.predict(X_test_dtm.toarray())')

from sklearn.metrics import accuracy_score
print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_pred_class)*100, 2)))

# Confusion Matrix
from sklearn.metrics import plot_confusion_matrix
maxtix = plot_confusion_matrix(nb, X_test_dtm.toarray(), y_test, cmap='Blues', values_format='.3g')

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class, digits=4))


# #### Features Not Contain Stop Words

# import and instantiate CountVectorizer (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_stop = CountVectorizer(stop_words = "english")
vect_stop

get_ipython().run_cell_magic('time', '', "# create a document-term matrix from all of the training data\nX_train_dtm = vect_stop.fit_transform(X_train)\nprint('X_train_dtm shape: {}'.format(X_train_dtm.shape))\nX_test_dtm = vect_stop.transform(X_test)\nprint('X_test_dtm shape: {}'.format(X_test_dtm.shape))")

print('There are {} tokens'.format(len(vect_stop.get_feature_names())))
print(vect_stop.get_feature_names()[:50])

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# Train the model and make a prediction\nnb.fit(X_train_dtm.toarray(), y_train)\ny_pred_class = nb.predict(X_test_dtm.toarray())')

from sklearn.metrics import accuracy_score
print('Accuracy: {}%'.format(round(accuracy_score(y_test, y_pred_class)*100, 2)))

# Confusion Matrix
from sklearn.metrics import plot_confusion_matrix
maxtix = plot_confusion_matrix(nb, X_test_dtm.toarray(), y_test, cmap='Blues', values_format='.3g')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_class)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class, digits=4))

y_test.value_counts()

reviews.loc[[59, 73, 44, 56, 74, 81, 84, 53, 54, 39, 40, 31, 80, 27, 38, 55, 10,
             2, 91, 89, 48, 19, 86], 'lie']

y_test

result = (y_test.value_counts()[0]/y_test.shape[0])
print('Base model(null model) for lie prediction in testing set: {}%'.format(round(result*100, 2)))


# #### K-fold Cross-validation

import numpy as np
np.random.seed(1)
print('Base model(null model) for lie prediction: {}%'.format((y.value_counts()/y.shape[0])[0]))


# 5 folds
get_ipython().run_cell_magic('time', '', 'nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words = "english")\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=5, scoring=\'accuracy\').mean()\nprint(\'Accuracy: {}%\'.format(round(result*100, 2)))')

# The 5-fold cross-validation model has 59.71 accuracy rate which is higher than the base model. 
get_ipython().run_cell_magic('time', '', 'nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words = "english")\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=5, scoring=\'recall\').mean()\nprint(\'Recall: {}%\'.format(round(result*100, 2)))')


# 10 folds

get_ipython().run_cell_magic('time', '', 'nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words = "english")\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=10, scoring=\'accuracy\').mean()\nprint(\'Accuracy: {}%\'.format(round(result*100, 2)))')

get_ipython().run_cell_magic('time', '', 'nb = MultinomialNB()\n\nvect_cv= CountVectorizer(stop_words = "english")\nX_dtm = vect_cv.fit_transform(X)\n\n# evaluate with 5-fold cross-validation (using X instead of X_train)\nfrom sklearn.model_selection import cross_val_score\nresult = cross_val_score(nb, X_dtm.toarray(), y, cv=10, scoring=\'recall\').mean()\nprint(\'Recall: {}%\'.format(round(result*100, 2)))')


# #### Feature Ranking in MultinomialNB

X_tokens = vect_cv.get_feature_names()

# examine the first 50 tokens
print(X_tokens[0:50])

# examine the last 50 tokens
print(X_tokens[-50:])

# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

get_ipython().run_cell_magic('time', '', '# train the model using X_dtm\nnb.fit(X_dtm, y)')

nb.feature_count_
# rows represent classes, columns represent tokens
print("Shape: {}".format(nb.feature_count_.shape))

# number of times each token appears across all 'fake' class
fake_token_count = nb.feature_count_[0, :]
fake_token_count

# number of times each token appears across all 'true' class
true_token_count = nb.feature_count_[1, :]
true_token_count

# create a DataFrame of tokens with their separate atheism, forsale and med counts
tokens = pd.DataFrame({'token':X_tokens, 'fake':fake_token_count, 
                       'true':true_token_count}).set_index('token')

# examine 5 random DataFrame rows
tokens.sample(5, random_state=6)

# Naive Bayes counts the number of observations in each class
nb.class_count_

# Naive Bayes counts the number of observations for all classes
Total_vocab = tokens.shape[0]
print('Number of vocabulary learned: {}'.format(Total_vocab))

# calculate the condition probabilities using Laplace smoother (N + |vocab|)
# https://scikit-learn.org/stable/modules/naive_bayes.html
tokens['fake'] = (tokens.fake + 1) / (nb.class_count_[0]+Total_vocab)
tokens['true'] = (tokens.true + 1) / (nb.class_count_[1]+Total_vocab)
tokens['fake_ratio'] = tokens.fake/tokens.true
tokens['true_ratio']=tokens.true/tokens.fake
tokens = tokens.sort_values('fake_ratio', ascending=False)
tokens.head(20)

print('most indicative words for fake review')
tokens.loc[:,'fake'].sort_values(ascending=False)[:20]

print('most indicative words for true review')
tokens.loc[:,'true'].sort_values(ascending=False)[:20]

