#!/usr/bin/env python
# coding: utf-8

# Compare MNB and SVMs 
# 
# 
# for (Kaggle Sentiment Classification – Kaggle is optional and you may find, gather with an API, or create any good sentiment type data set)  
# 
# 1)	Be sure to use the Assignment format. Remember that the dataset defines the topic/area.  
# 
# So far we have learned how to use sklearn to build MNB and SVMs models and evaluate them using various test methods and measures.   
# 
# Now consult the sklearn documentation   
# 
# and revise the instructor's given script  (using the instructor script is optional and you can also use my code. Ideally – you should practice writing your own code for SVM and NB).  
# 
# Your Output/Results should include visualizations, tables, confusion matrices, etc.   
# Include precision and recall.  
# 
# to output confusion matrix, precision and recall values for the Kaggle Sentiment training data. Remember the sample script used 60% data for training and 40% for testing.   
# 
# While 60/40 is fine for testing and training, you are free to use 30/70 and/or to try both and compare.   
# 
# As part of this assignment, try to use both CountVectorizer AND TfidfVectorizer for the data and the apply both to your Naïve Bayes (MNB) and your SVM.   
# 
# Required: For the SVM – use three kernels and also try a few different costs for each kernel. Create a table to compare.
# 
# Also be sure to compare the accuracies of MNB and SVM.  
# 
# -	Try to find a good method to generate the top 10 indicative words for the most positive category and the most negative category from the MNB and SVMs models respectively. You can find code for this that I posted  and in the asynch. You can also Google it. 
# 


import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import random


random.seed(2)
pd.set_option('display.max_colwidth', None)


with open('./data/opinion-lexicon-English/positive-words.txt', 'r') as f:
    pos_words = f.readlines()
pos_words = [re.sub(r'\n','', w) for w in pos_words[31:]]

with open('./data/opinion-lexicon-English/negative-words.txt', 'r', encoding='ISO-8859-1') as f:
    neg_words = f.readlines()
neg_words = [re.sub(r'\n','', w) for w in neg_words[31:]]


reviews = pd.read_csv('./data/train.csv', index_col=0)
reviews.sample(5)

print('Number of documents: {}'.format(reviews.shape[0]))



reviews['str_length'] = reviews.reviews.apply(lambda string: len(string))
reviews['n_tokens'] = reviews.reviews.apply(lambda string: len(string.split()))
reviews['tokens_avg_len'] = reviews.reviews.apply(lambda string: round(np.mean([len(t) for t in string.split()]), 2))
reviews['exclamat'] = reviews.reviews.apply(lambda string: np.sum([1 if l == '!' else 0 for l in string]))
reviews['question'] = reviews.reviews.apply(lambda string: np.sum([1 if l == '?' else 0 for l in string]))
reviews['plus'] = reviews.reviews.apply(lambda string: np.sum([1 if l == '+' else 0 for l in string]))
reviews['period'] = reviews.reviews.apply(lambda string: np.sum([1 if l == '.' else 0 for l in string]))
reviews['minus'] = reviews.reviews.apply(lambda string: np.sum([1 if l == '-' else 0 for l in string]))

# it tooks about 8min 12s for looping through the list of good words and bad words
reviews['n_pos'] = reviews.reviews.apply(lambda st: np.sum([1 if s.lower() in pos_words else 0 for s in st.split()]))
reviews['n_neg'] = reviews.reviews.apply(lambda st: np.sum([1 if s.lower() in neg_words else 0 for s in st.split()]))


# ## EDA
reviews.label.value_counts().plot(kind='bar', figsize=(8,6), fontsize=11)
plt.title('Raw Dataset', size=12)
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.xticks((0,1),['pos', 'neg'])
# add annotation on each bar
for i in range(2):
    plt.text(x = i - 0.05, y = reviews.label.value_counts().values[i] + max(reviews.label.value_counts())/100, 
             s = reviews.label.value_counts().values[i])
plt.show()


reviews.groupby('label').str_length.mean()


reviews.groupby('label').n_tokens.mean()


reviews.boxplot(column=['str_length', 'n_tokens'], by=['label'])
plt.show()
# `str_length` and `n_tokens` has small distinctions


reviews.groupby('label').tokens_avg_len.mean()

reviews.groupby('label').exclamat.mean()

reviews.groupby('label').plus.mean()

reviews.groupby('label').period.mean()

reviews.groupby('label').minus.mean()

reviews.groupby('label').n_pos.mean()

reviews.groupby('label').n_neg.mean()

reviews.boxplot(column=['n_pos', 'n_neg'], by=['label'])
plt.show()
# `n_pos` and `n_neg` has distinctions



# ## Model
# define X and y
X = reviews.reviews
y = reviews.label
# check the shape
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

reviews.head()


## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize

#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 

from nltk.stem.porter import PorterStemmer
STEMMER=PorterStemmer()
# print(STEMMER.stem("singer"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words


# ### Pipeline with vectorizor features
print('Base model(null model) for sentiment prediction: {}%'.format((y.value_counts()/y.shape[0])[0]))


def top_bottom_10_features(clf, vect):
    '''
    expect: a classifier/pipline step, a vectorizer
    modify: get the top and bottom 10 featurs from coef_.ravel()
    return: print output
    '''
    coef = clf.coef_.ravel()
    feature_names = np.array(vect.get_feature_names())
    top_feat_coef_idx = np.argsort(coef)[-10:][::-1] # np.argsort retrurn index by the values in increasing order
    top_feat_words = feature_names[top_feat_coef_idx]
    bot_feat_coef_idx = np.argsort(coef)[:10]
    bot_feat_words = feature_names[bot_feat_coef_idx]
    print('\nTop 10 feature words:')
    for i in range(len(top_feat_words)):
        print('{:15} : {}'.format(top_feat_words[i], coef[top_feat_coef_idx[i]]))

    print('\nBottom 10 feature words:')
    for i in range(len(bot_feat_words)):
        print('{:15} : {}'.format(bot_feat_words[i], coef[bot_feat_coef_idx[i]]))



def top_bottom_feature_plot(clf, vect, n_features=10):
    '''
    expect: a classifier/pipline step, a vectorizer, and a number of features (default = 10)
    modify: barplot of the top_features of the features
    return: a barplot
    '''
    import matplotlib.pyplot as plt
    coef = clf.coef_.ravel()  # ravel() flatten an array
    top_positive_coef = np.argsort(coef)[-n_features:] # np.argsort retrurn index by the values in increasing order
    top_negative_coef = np.argsort(coef)[:n_features]
    top_coef = np.hstack([top_negative_coef, top_positive_coef])
    plt.figure(figsize=(14, 8))
    # encode color: below the median is red; otherwise is blue
    colors = ['red' if c < list(coef)[int((len(coef)/2)+1)] else 'blue' for c in coef[top_coef]]
    plt.bar(x = np.arange(2 * n_features), height = coef[top_coef], edgecolor='black',
            color=colors)
    feature_names = np.array(vect.get_feature_names())
    plt.xticks(np.arange(1 + 2 * n_features), feature_names[top_coef], rotation=60, ha='right')
    plt.title('Top 10 and Bottom 10 Feature Words')
    plt.show()



def make_prediction(X, y, vect, clf):
    '''
    X: pandas series of text
    y: pandas series of label
    vect: an instantiated vectorizer
    clf: an instantiated classifier
    Workflow: 1. train_test_split for holdout test
              2. pipe = vectorizer + classifier
              3. CV on pipe with X_train
              4. Validate with X_test on pipe.predict
              5. Plot and show the information of top 10 and bottom 10 features
              6. Confusion Matrix and Classification Report
    '''
    # info of vect and clf
    print(vect)
    print(clf)
    
    # 1. Split data for holdout set
    from sklearn.model_selection import train_test_split # test_size=0.25 in default; stratify in default for same distrtubution
    # set test_size=0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
        
        
    # 2. Create a pipeline of Vectoerizer and classifier
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(vect, clf)
    
    
    # 3. CV on pipe with X_train
    from sklearn.model_selection import cross_val_score
    train_accuracy= cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
    print('\nTraining Accuracy with 5-fold CV: {}%'.format(round(train_accuracy*100, 2)))
    
    
    # 4. Validate with X_test on pipe.predict
    pipe.fit(X_train, y_train)
    y_pred_class = pipe.predict(X_test)
    from sklearn import metrics
    test_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('\nTesting Accuracy: {}%'.format(round(test_accuracy*100, 2))) 
    

    # 5. Plot and show the information of top 10 and bottom 10 features
    ## print the number of vocaburary learned during the training
    print('\nNumber of features: {}\n'.format(len(vect.get_feature_names())))
    if (list(pipe.named_steps.keys())[1] == 'multinomialnb') or (list(pipe.named_steps.keys())[1] == 'linearsvc'):
        top_bottom_feature_plot(pipe.steps[1][1], vect, n_features=10) 
        top_bottom_10_features(pipe.steps[1][1], vect)
    
    
    # 6. Confusion Matrix and Classification Report
    ## Confusion Matrix
    from sklearn.metrics import plot_confusion_matrix
    matrix = plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format='.3g')
    print("\nConfusion Matrix:")
    plt.show()
    
    ## Classification Report
    from sklearn.metrics import classification_report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_class, digits=4))


# #### NB + CountVect

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y,vect ,nb)



# #### NB + CountVect(binry)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, binary=True, stop_words='english')

make_prediction(X, y,vect ,nb)


# #### NB + Tfidf

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### linearSVM(C=10) + CountVect(bigrams)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=10)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english', ngram_range=(1, 2))

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=1) + CountVect(bigrams)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=1)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english', ngram_range=(1, 2))

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=10) + CountVect(trigrams)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=10)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english', ngram_range=(1, 3))

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=1) + CountVect(trigrams)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=1)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english', ngram_range=(1, 3))

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=10) + CountVect

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=10)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=10) + CountVect(binary)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=10)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, binary=True, stop_words='english')

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=1) + CountVect(binary)

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=1)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, binary=True, stop_words='english')

make_prediction(X, y, vect ,svm)


# #### linearSVM(C=1) + CountVect

# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=1)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, vect ,svm)


# #### SVM (rbh) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 10, kernel='rbf')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (rbh, C=1) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 1, kernel='rbf')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (rbh, C=10) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 10, kernel='rbf')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, vect ,svm_rbf)


# #### SVM (rbh, C=1) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 1, kernel='rbf')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, vect ,svm_rbf)


# #### SVM (poly) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 10, kernel='poly')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (poly, C=1) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 1, kernel='poly')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (poly, c=10) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C=10, kernel='poly')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y,vect ,svm_rbf)


# #### SVM (poly, C=1) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C=1, kernel='poly')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y,vect ,svm_rbf)


# #### SVM (sigmoid) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 10, kernel='sigmoid')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (sigmoid, C=1) + Tfidf

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C = 1, kernel='sigmoid')

# import and instantiate CountVetorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5, stop_words='english')

make_prediction(X, y, tfidf ,nb)


# #### SVM (sigmoid, C=10) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C=10, kernel='sigmoid')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y,vect ,svm_rbf)


# #### SVM (sigmoid, C=1) + CountVect

# import and instantiate SVC (with the soft margin/cost C = 10, and kernel='rbf')
from sklearn.svm import SVC 
svm_rbf = SVC(C=1, kernel='sigmoid')

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english')

make_prediction(X, y,vect ,svm_rbf)



# ## Adding features to a document-term matrix (using `FeatureUnion`)
# 
# - Below is a process that does allow for proper cross-validation, and does integrate well with the scikit-learn workflow.
# - To use this process, we have to learn about transformers, **`FunctionTransformer`**, and **`FeatureUnion`**.
# 
# Transformer objects provide a `transform` method in order to perform **data transformations**. Here are a few examples:
# 
# - **`CountVectorizer`**
#     - `fit` learns the vocabulary
#     - `transform` creates a document-term matrix using the vocabulary
# - **`SimpleImputer`**
#     - `fit` learns the value to impute
#     - `transform` fills in missing entries using the imputation value
# - **`StandardScaler`**
#     - `fit` learns the mean and scale of each feature
#     - `transform` standardizes the features using the mean and scale
# - **`HashingVectorizer`**
#     - `fit` is not used, and thus it is known as a "stateless" transformer
#     - `transform` creates the document-term matrix using a hash of the token
#     
#  [FunctionTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
#  
# ![Pipeline versus FeatureUnion](transformer_example.png)
# 
# ![Transformation_Workflow](transformation_workflow.png)




# define X and y
X = reviews.loc[:, ['reviews', 'n_pos' ,'n_neg', 'str_length', 'n_tokens']] # put necessary features
y = reviews.label



# define a function that accepts a DataFrame returns the manually created features
def get_manual(df):
    return df.loc[:, ['n_pos' ,'n_neg','str_length', 'n_tokens']]

from sklearn.preprocessing import FunctionTransformer
# create a stateless transformer from the get_manual function
get_manual_ft = FunctionTransformer(get_manual, validate=False)



# define a function that accepts a DataFrame returns the ingredients string
def get_text(df):
    return df.reviews

# create another transformer
get_text_ft = FunctionTransformer(get_text, validate=False)


# #### Combining feature extraction steps

# With the `FunctionTransformer` built, it can be combined into pipeline using `make_pipleine` and `make_union`.  
#   * `make_union` can contain `make_pipleine` and `FunctionTransformer`.  
#       * `make_pipleine` can contain `FunctionTransformer` and `CountVectorizer` as a series of pipeline




def make_prediction_manual_ft(X, y, vect, clf):
    '''
    X: pandas series of text
    y: pandas series of label
    vect: an instantiated vectorizer
    clf: an instantiated classifier
    Workflow: 1. train_test_split for holdout test
              2. pipe = [(get_text_ft + vectorizer) + get_manual_ft]+ clf
              3. CV on pipe with X_train
              4. Validate with X_test on pipe.predict
              5. Confusion Matrix and Classification Report
    '''
    # info of vect and clf
    print(vect)
    print(clf)
    
    # 1. Split data for holdout set
    from sklearn.model_selection import train_test_split # test_size=0.25 in default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        
        
    # 2. Create a complete pipeline
    ## 2.1 make union: (get_text_ft + vectorizer) + get_manual_ft
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import make_union
    union = make_union(make_pipeline(get_text_ft, vect), get_manual_ft)

    ## 2.2 make pipleine: [union]+ clf
    pipe = make_pipeline(union, clf)
    
    
    # 3. CV on pipe with X_train
    ## properly cross-validate the entire pipeline (and pass it the entire DataFrame)
    from sklearn.model_selection import cross_val_score
    train_accuracy= cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
    print('\nTraining Accuracy with 5-fold CV: {}%'.format(round(train_accuracy*100, 2)))

    
    # 4. Validate with X_test on pipe.predict
    pipe.fit(X_train, y_train)
    y_pred_class = pipe.predict(X_test)
    from sklearn import metrics
    test_accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('\nTesting Accuracy: {}%'.format(round(test_accuracy*100, 2)))

    
    # 5. Confusion Matrix and Classification Report
    ## Confusion Matrix
    from sklearn.metrics import plot_confusion_matrix
    matrix = plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues', values_format='.3g')
    print("\nConfusion Matrix:")
    plt.show()
    
    ## Classification Report
    from sklearn.metrics import classification_report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_class, digits=4))




# import and instantiate LinearSVC (with the soft margin/cost C = 10)
from sklearn.svm import LinearSVC 
svm = LinearSVC(C=1)

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5, stop_words='english', ngram_range=(1, 3))

make_prediction_manual_ft(X, y,vect ,svm)


