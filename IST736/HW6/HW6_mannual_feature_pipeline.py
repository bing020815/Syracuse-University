#!/usr/bin/env python
# coding: utf-8

# # HW6 
# 
# 1)	Use the required format for this assignment  
# 2)	For this assignment you will use the Benoulli (so here – be sure to use the binary=True in the CountVectorizer).  So, you will need to create two dataframes – one for Bernoulli and one for the normal frequency counts. For those who want to do more, you can also use the TfidfVectorizer and create a third dataframe that is normalized. This is fast to do and will create an interesting opportunity to compare, binary, count, and tf-idf normalized. However, it is optional.   
# 3)	Next, on the frequency data frame (and on the tfidf if you create one) use the Multinomial Naïve Bayes algorithms   
# 4)	Datasets:  
# a.	First – create your own small, easy, predictable dataset (as either csv or corpus – your choice). Code the above to assure that you code works. You may choose to share this step in the Analysis section under models and methods.    
# b.	Next, use any labeled dataset (50 rows or more so you get interesting results). You can use csv or corpus. You can use the lie data or the sentiment data from class (and can repurpose any intros that you wrote). You can also find or use and API to get data. There are many levels here and it depends on how advanced you want to go. While it is not required, it would be great for you to try to use an API, clean up the data, and then use the methods on it.    
# 
# Remember that the Benoulli model takes Boolean vectors as input, 
# 
# NOTE: A Boolean Vector is the same as *binary* data. This means that instead of each word (in each column) being counted up for each document, rather, there is a 0 if the word is NOT in the doc and 1 if it is. 
# 
# and the Multinomial model takes frequency vectors as input. 
# 
# NOTE: This means that it uses the normal counts (and also can work on tf-idf or other normalized counts).
# 
# NOTE: You will need to create a Training Set and a Testing Set AFTER you clean up and fully prep your data. Remember that Naïve Bayes (as well as Bernoulli) needs to be trained first and then tested. In both cases REMOVE AND KEEP the labels 
# 
# 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 1000)

reviews = pd.read_csv('deception_data_converted_final.tsv', sep='\t')
reviews.shape

reviews.sample(5, random_state=5)


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
lie_mapper = {'f': 1, 't': 0} # true = 0 ; fake = 1
reviews['review'] = reviews.review.map(clean_str)
reviews['lie_num'] = reviews.lie.map(lie_mapper)
reviews['sentiment_num'] = reviews.sentiment.map(sentiment_mapper)
reviews['length'] = reviews.review.apply(lambda string: len(string))
reviews['n_tokens'] = reviews.review.apply(lambda string: len(string.split()))
reviews['exclamat'] = reviews.review.apply(lambda string: 1 if '!' in string else 0) 
reviews['question'] = reviews.review.apply(lambda string: 1 if '?' in string else 0) 
reviews['neg_long_len'] = reviews.apply(lambda row: 1 if (row.sentiment=='n')&(row.length>400) else 0, axis = 1)
reviews['pos_less_token'] = reviews.apply(lambda row: 1 if (row.sentiment=='p')&(row.n_tokens<50) else 0, axis = 1)


reviews[(reviews.sentiment=='n')&(reviews.length>400)].loc[:,['lie', 'sentiment', 'length']].lie.value_counts()

reviews[(reviews.sentiment=='n')&(reviews.length>400)].loc[:,['lie', 'sentiment', 'length']]



# ## EDA

# #### Sentiment

reviews.sentiment.value_counts().plot(kind='bar', figsize=(8,6), fontsize=11)
plt.title('Raw Dataset', size=12)
plt.xlabel('Sentiment')
plt.ylabel('Counts')
# add annotation on each bar
for i in range(2):
    plt.text(x = i - 0.03 , y = reviews.sentiment.value_counts().values[i] + 0.4, 
             s = reviews.sentiment.value_counts().values[i])
plt.show()

reviews.boxplot(column=['length', 'n_tokens'], by=['sentiment'])
plt.show()

reviews.boxplot(column=['exclamat', 'question'], by=['sentiment'])
plt.show()

# `'length'`,`'exclamat'` have distinction between two classes

# #### Lie

reviews.lie.value_counts().plot(kind='bar', figsize=(8,6), fontsize=11)
plt.title('Raw Dataset', size=12)
plt.xlabel('lie')
plt.ylabel('Counts')
# add annotation on each bar
for i in range(2):
    plt.text(x = i - 0.03 , y = reviews.lie.value_counts().values[i] + 0.4, 
             s = reviews.lie.value_counts().values[i])
plt.show()

reviews.boxplot(column=['length', 'n_tokens'], by=['lie'])
plt.show()

reviews.boxplot(column=['exclamat','question'], by=['lie'])
plt.show()

reviews.boxplot(column=['neg_long_len', 'pos_less_token'], by=['lie'])
plt.show()


# `'neg_long_len'`,`'exclamat'` have distinction between two classes



# ## Model - Sentiment

# define X and y
X = reviews.review
y = reviews.sentiment_num
# check the shape
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')


# ### Pipeline with vectorizor features

print('Base model(null model) for sentiment prediction: {}%'.format((y.value_counts()/y.shape[0])[0]))


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
              5. Confusion Matrix and Classification Report
    '''
    # info of vect and clf
    print(vect)
    print(clf)
    
    # 1. Split data for holdout set
    from sklearn.model_selection import train_test_split # test_size=0.25 in default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        
        
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


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(min_df=5, binary=True)

make_prediction(X, y,vect_b ,bnb)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5)

make_prediction(X, y,vect ,nb)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5)

make_prediction(X, y, tfidf ,nb)


# ## Efficiently searching for tuning parameters using RandomizedSearchCV
# 
# - When there are many parameters to tune, searching all possible combinations of parameter values may be **computationally infeasible**.
# - **`RandomizedSearchCV`** searches a sample of the parameter values, and you control the computational "budget".
# 
# [RandomizedSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# 
# [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

# ### 1. Rebuild the pipeline combination

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(tfidf, nb)
pipe.steps

# pipeline steps are automatically assigned names by make_pipeline
pipe.named_steps.keys()


# ### 2. Random Grid Search with CV

import scipy as sp
# set a random seed for sp.stats.uniform
import numpy as np
np.random.seed(1)

from sklearn.model_selection import RandomizedSearchCV
# for any continuous parameters, specify a distribution instead of a list of options
param_grid = {}
param_grid['tfidfvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-zA-Z ]+)'", r'(?u)\b\w\w+\b']
param_grid['tfidfvectorizer__ngram_range'] = [(1, 1),(1, 2)]
param_grid['tfidfvectorizer__min_df'] = [1,3,5]
param_grid['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
param_grid

def grid_result(pipe, param_grid):
    '''
    '''
    from sklearn.model_selection import RandomizedSearchCV
    # additional parameters are n_iter (number of searches) and random_state
    rand = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_iter=5, random_state=1)   
    get_ipython().run_line_magic('time', 'rand.fit(X, y)')
    # time the randomized search
    
    print('Random Search Best Score: \n  {}%\n'.format(round(rand.best_score_*100, 2)))
    print('Random Search Best Parameters: \n{}\n'.format(rand.best_params_))
    
    results = pd.DataFrame(rand.cv_results_)
    return results[['mean_test_score', 'std_test_score', 'params']]

grid_result(pipe, param_grid)


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
X = reviews.loc[:, ['review','length']] # put necessary features
y = reviews.sentiment_num


# ##### FunctionTransformer 1

# define a function that accepts a DataFrame returns the manually created features
def get_manual(df):
    return df.loc[:, ['length']]

from sklearn.preprocessing import FunctionTransformer
# create a stateless transformer from the get_manual function
get_manual_ft = FunctionTransformer(get_manual, validate=False)


# ##### FunctionTransformer 2

# define a function that accepts a DataFrame returns the ingredients string
def get_text(df):
    return df.review

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


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(binary=True)

make_prediction_manual_ft(X, y,vect_b ,bnb)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

make_prediction_manual_ft(X, y,vect ,nb)


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

make_prediction_manual_ft(X, y, tfidf ,nb)


# ## Efficiently searching for tuning parameters using RandomizedSearchCV
# 
# - When there are many parameters to tune, searching all possible combinations of parameter values may be **computationally infeasible**.
# - **`RandomizedSearchCV`** searches a sample of the parameter values, and you control the computational "budget".
# 
# [RandomizedSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# 
# [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

# ### 1. Rebuild the pipeline combination

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

# 2. Create a complete pipeline
## 2.1 make union: (get_text_ft + vectorizer) + get_manual_ft
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
union = make_union(make_pipeline(get_text_ft, vect), get_manual_ft)

## 2.2 make pipleine: [union]+ clf
pipe = make_pipeline(union, nb)
pipe.steps

# pipeline steps are automatically assigned names by make_pipeline
pipe.named_steps.keys()


# ### 2. Random Grid Search with CV

import scipy as sp
# set a random seed for sp.stats.uniform
import numpy as np
np.random.seed(1)

from sklearn.model_selection import RandomizedSearchCV
# for any continuous parameters, specify a distribution instead of a list of options
param_grid = {}
param_grid['featureunion__pipeline__countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-zA-Z ]+)'", r'(?u)\b\w\w+\b']
param_grid['featureunion__pipeline__countvectorizer__min_df'] = [1, 5]
param_grid['multinomialnb__alpha'] = sp.stats.uniform(scale=1)
param_grid

# def grid_result_manual_ft(pipe, param_grid):
#     '''
    
#     '''
#     from sklearn.model_selection import RandomizedSearchCV
#     # additional parameters are n_iter (number of searches) and random_state
#     rand = RandomizedSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_iter=5, random_state=1)   
#     %time rand.fit(X, y) 
#     # time the randomized search
    
#     print('Random Search Best Score: \n  {}%\n'.format(round(rand.best_score_*100, 2)))
#     print('Random Search Best Parameters: \n{}\n'.format(rand.best_params_))
    
#     results = pd.DataFrame(rand.cv_results_)
#     return results[['mean_test_score', 'std_test_score', 'params']]

grid_result(pipe, param_grid)

#    

#    

#    

# 

#    

#    

#    

#    

#    

#    

#    

# ## Model - Lie

# define X and y
X = reviews.review
y = reviews.lie_num
# check the shape
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

print('Base model(null model) for sentiment prediction: {}%'.format((y.value_counts()/y.shape[0])[0]))

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(binary=True)

make_prediction(X, y,vect_b ,bnb)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

make_prediction(X, y,vect ,nb)


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

make_prediction(X, y, tfidf ,nb)


# ## Efficiently searching for tuning parameters using RandomizedSearchCV
# 
# - When there are many parameters to tune, searching all possible combinations of parameter values may be **computationally infeasible**.
# - **`RandomizedSearchCV`** searches a sample of the parameter values, and you control the computational "budget".
# 
# [RandomizedSearchCV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# 
# [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

# ### 1. Rebuild the pipeline combination


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(binary=True)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vect_b, bnb)
pipe.steps

# pipeline steps are automatically assigned names by make_pipeline
pipe.named_steps.keys()


# ### 2. Random Grid Search with CV


import scipy as sp
# set a random seed for sp.stats.uniform
import numpy as np
np.random.seed(1)

from sklearn.model_selection import RandomizedSearchCV
# for any continuous parameters, specify a distribution instead of a list of options
param_grid = {}
param_grid['countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-zA-Z ]+)'", r'(?u)\b\w\w+\b']
param_grid['countvectorizer__ngram_range'] = [(1, 1),(1, 2)]
param_grid['countvectorizer__min_df'] = [1,3,5]
param_grid['bernoullinb__alpha'] = sp.stats.uniform(scale=1)
param_grid

grid_result(pipe, param_grid)


#   

#   

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
X = reviews.loc[:, ['review','neg_long_len', 'exclamat']] # put necessary features
y = reviews.lie_num


# ##### FunctionTransformer 1

# define a function that accepts a DataFrame returns the manually created features
def get_manual(df):
    return df.loc[:, ['neg_long_len', 'exclamat']]

from sklearn.preprocessing import FunctionTransformer
# create a stateless transformer from the get_manual function
get_manual_ft = FunctionTransformer(get_manual, validate=False)


# ##### FunctionTransformer 2


# define a function that accepts a DataFrame returns the ingredients string
def get_text(df):
    return df.review

# create another transformer
get_text_ft = FunctionTransformer(get_text, validate=False)


# #### Combining feature extraction steps

# With the `FunctionTransformer` built, it can be combined into pipeline using `make_pipleine` and `make_union`.  
#   * `make_union` can contain `make_pipleine` and `FunctionTransformer`.  
#       * `make_pipleine` can contain `FunctionTransformer` and `CountVectorizer` as a series of pipeline


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(binary=True)

make_prediction_manual_ft(X, y,vect_b ,bnb)

# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer (with min_df=5)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

make_prediction_manual_ft(X, y,vect ,nb)


# import and instantiate Multinomial Naive Bayes (with the default parameters)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

make_prediction_manual_ft(X, y, tfidf ,nb)


# ## Efficiently searching for tuning parameters using RandomizedSearchCV

# ### 1. Rebuild the pipeline combination

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vect_b, bnb)

# import and instantiate Bernoulli Naive Bayes (with the default parameters)
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

# import and instantiate CountVectorizer for BernoulliNB (with default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect_b = CountVectorizer(binary=True)

# 2. Create a complete pipeline
## 2.1 make union: (get_text_ft + vectorizer) + get_manual_ft
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
union = make_union(make_pipeline(get_text_ft, vect_b), get_manual_ft)

## 2.2 make pipleine: [union]+ clf
pipe = make_pipeline(union, bnb)
pipe.steps


# pipeline steps are automatically assigned names by make_pipeline
pipe.named_steps.keys()


# ### 2. Random Grid Search with CV

import scipy as sp
# set a random seed for sp.stats.uniform
import numpy as np
np.random.seed(1)

from sklearn.model_selection import RandomizedSearchCV
# for any continuous parameters, specify a distribution instead of a list of options
param_grid = {}
param_grid['featureunion__pipeline__countvectorizer__token_pattern'] = [r"\b\w\w+\b", r"'([a-zA-Z ]+)'", r'(?u)\b\w\w+\b']
param_grid['bernoullinb__alpha'] = sp.stats.uniform(scale=1)
param_grid

grid_result(pipe, param_grid)

