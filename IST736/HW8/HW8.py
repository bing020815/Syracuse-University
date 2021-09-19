#!/usr/bin/env python
# coding: utf-8

# Topic Modeling
# 
# 
# 
# LDA is an algorithm that can “summarize” the main topics of a text collection, now you are asked to use this algorithm to analyze the main topics in the floor debate of the 110th Congress (House only). According to political scientists, there are usually 40-50 common topics going on in each Congress. Tune the number of topics and see if LDA can get you the common topics, such as defense, education, healthcare, economy, etc.
# 
# 
# The data set “110” consists of four subfolders. For the subfolder names, “m” means “male”, “f” means “female”, “d” means “democrat”, “r” means “republican”. You can merge all of them into one folder to run Mallet LDA.
# 
# OR If you cannot work with Mallet – you can use my code for LDA which also generates a great interactive vis. 
# 
# Here is my folder filled with many LDA and API examples:
# https://drive.google.com/drive/folders/1_QMxLIffDshlY8U2Nn_yJlAY_5ToF_0e?usp=sharing
# 
# Again – you do not need to use Mallet – you may if you wish. You may also use both or multiple methods if you want to go deep.
# 
# There are a few other parameters you can tune, such as ngram (for Mallet only). You can decide what parameters to use and explain your decision in the report.
#  
# Interpreting topic clustering results is very difficult. See if this article “Reading Tea Leaves” may help you. http://www.umiacs.umd.edu/~jbg/docs/nips2009-rtl.pdf. The recommended readings are also great examples to demonstrate how to articulate topic modeling results. 
# 
# This is a fairly large data set (100M pure text, more than 400 files). Please start working on it early because it may take a long time to run. 
# 
# ALSO – do not start with this HUGE dataset. First, create a small and balanced sample of data or even a different dataset that you find or make that will work. Make sure your code runs on that smaller data first. Then, when everything work – try it on the large dataset. If you still cannot – cut the large dataset down until you can.
# 
# 
# 
# NOTE from other Professor: 
# To prevent your program from being interrupted, run it as a backend process by adding "&" to the end of your command (for Linux system). Or you can use one subset of the data to build a topic model and explain what topics you have discovered from the data.


import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('colheader_justify', 'center')





# use os to create a list of file name
import os
f_d_path = r'./110/110-f-d'
f_r_path = r'./110/110-f-r'
m_d_path = r'./110/110-m-d'
m_r_path = r'./110/110-m-r'

# # assgin the list of filename to variables
f_d_FileNameList=os.listdir(f_d_path)
f_r_FileNameList=os.listdir(f_r_path)
m_d_FileNameList=os.listdir(m_d_path)
m_r_FileNameList=os.listdir(m_r_path)


# use glob to create a list of file path with file name
import glob
f_d_filenames = glob.glob('./110/110-f-d/*.txt')
f_r_filenames = glob.glob('./110/110-f-r/*.txt')
m_d_filenames = glob.glob('./110/110-m-d/*.txt')
m_r_filenames = glob.glob('./110/110-m-r/*.txt')

print("\nf_d_filenames_path:")
print(f_d_filenames[:10])
print("\nf_r_filenames_path:")
print(f_r_filenames[:10])
print("\nm_d_filenames_path:")
print(m_d_filenames[:10])
print("\nm_r_filenames_path:")
print(m_r_filenames[:10])


print(f'f_d: {len(f_d_filenames)}')
print(f'f_r: {len(f_r_filenames)}')
print(f'm_d: {len(m_d_filenames)}')
print(f"m_r: {len(m_r_filenames)}")


# read the contents of the f_d files into a list (each list element is one txt)
# https://medium.com/@apiltamang/unicode-utf-8-and-ascii-encodings-made-easy-5bfbe3a1c45a
# unicode: utf-8, ascii, latin
f_d_text = []
for filename in f_d_filenames:
    with open(filename, encoding='latin') as f:
        f_d_text.append(f.read())
print("f_d_text:")
print(f_d_text[0])

# read the contents of the f_r files into a list (each list element is one txt)
# unicode: utf-8, ascii, latin
f_r_text = []
for filename in f_r_filenames:
    with open(filename, encoding='latin') as f:
        f_r_text.append(f.read())
print("f_r_text:")
print(f_r_text[0])

# read the contents of the m_d files into a list (each list element is one txt)
# unicode: utf-8, ascii, latin
m_d_text = []
for filename in m_d_filenames:
    with open(filename, encoding='latin') as f:
        m_d_text.append(f.read())
print("m_d_text:")
print(m_d_text[0])

# read the contents of the m_r files into a list (each list element is one txt)
# unicode: utf-8, ascii, latin
m_r_text = []
for filename in m_r_filenames:
    with open(filename, encoding='latin') as f:
        m_r_text.append(f.read())
print("m_r_text:")
print(m_r_text[0])



# combine the pos and neg for train lists
all_text = f_d_text + f_r_text + m_d_text + m_r_text
print(len(all_text))

# create a list of labels (pos=1, neg=0)
all_filename = f_d_FileNameList + f_r_FileNameList + m_d_FileNameList + m_r_FileNameList
print(len(all_filename))


import pandas as pd
# convert the lists into a DataFrame
df = pd.DataFrame({'filename':all_filename, 'text':all_text})
df.head()


type(df.text[0])



# Use regular express module to extract text content to perform LDA analysis.
# Target the `<TEXT>` and `</TEXT>` tag

def get_text_from_tags(string):
    """
    expect: a string
    modify: 1. extract the text inside the TEXT tags and return a list of text
                # https://regex101.com/r/BGpAT7/1
            2. join the list text to a string as a sigle document
            3. remove '\n', '``' and "''"
    return: return a string, doc
    """
    import re
    match_list = re.findall(r'<TEXT>\s+([\w\d\W]+?)\s+<\/TEXT>\s+', string)
    doc = ''.join(match_list)
    result = re.sub(r'\n', '', doc)
    result = re.sub(r'``', '', result)
    result = re.sub(r"''", '', result)
    return result


df['text'] = df.text.apply(get_text_from_tags)
df['filename'] = df.filename.apply(lambda string: string[4:-4])
df.head()



#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 

# Use NLTK's PorterStemmer in a function
def my_tokenizer_stemmer(str_input):
    import re
    from nltk.stem.porter import PorterStemmer
    stemmer=PorterStemmer()
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words

# tokenize the doc and lemmatize its tokens
def my_tokenizer_lemma(doc):
    import spacy
    # create a spaCy lemmatizer
    spacy.load('en_core_web_sm')
    lemmatizer = spacy.lang.en.English()
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])


# #### No stemming, default token_patterns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vect = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
vect.fit_transform(df.text)
vect.get_feature_names()

len(vect.get_feature_names())


# #### No stemming, only alphabet charactors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
pattern = r'(?u)\b[a-zA-Z][a-zA-Z]+\b'
# pattern = r'\b[a-zA-Z]{3,}\b'  ## raw string with word boundries, word at least 3 letter
vect = CountVectorizer(max_df=0.95,min_df=5, stop_words='english', token_pattern= pattern)
vect.fit_transform(df.text)
vect.get_feature_names()

len(vect.get_feature_names())


# ## CountVectorizer
# `CountVectorizer` can do:
# 1. build_preprocessor:
#     * Returns a `callable` to preprocess the input text vefore tokenization
# 2. build_tokenizer:
#     * Creates a function capable of splitting a document's corpus into tokens
# 3. build_analyzer:
#     * Builds a analyzer function which applies `preprocessing`, `tokenization`, remove `stop_words` and create `ngram_range`
#     * When a `customized analyzer` is used, `build_analyzer` method does not call `_word_ngrams`, which is responsible for removing __stop words__ and extracting __n-grams__
#     * One way to solve the issue above is to create custom vectorizer classes. 
#     * A new class inheriting from the base vectorizer and overwrite the `build_preprocessor`, `build_tokenizer` and/or `build_analyzer` methods as desired
#     
# 
# ### Defines a custom vectorizer class
# class CustomVectorizer(CountVectorizer): 
#     
#     # overwrite the build_analyzer method, allowing one to
#     # create a custom analyzer for the vectorizer
#     def build_analyzer(self):
#         
#         # load stop words using CountVectorizer's built in method
#         stop_words = self.get_stop_words()
#         
#         # create the analyzer that will be returned by this method
#         def analyser(doc):
#             
#             # load spaCy's model for english language
#             spacy.load('en_core_web_sm')
#             
#             # instantiate a spaCy tokenizer
#             lemmatizer = spacy.lang.en.English()
#             
#             # apply the preprocessing and tokenzation steps
#             import re
#             doc_clean = re.sub(r'[^\w]+|[\d,]+', ' ', doc)
#             tokens = lemmatizer(doc_clean)
#             lemmatized_tokens = [token.lemma_ for token in tokens]
#             
#             # use CountVectorizer's _word_ngrams built in method
#             # to remove stop words and extract n-grams
#             return(self._word_ngrams(lemmatized_tokens, stop_words))
#         return(analyser)
#     
# 
# custom_vec = CustomVectorizer(ngram_range=(1,2),
#                               stop_words='english')

# #### token_pattern = a-zA-Z_ and lemmatizer
## Customize stopwords list
from nltk.corpus import stopwords
my_stopwords = set(stopwords.words('english'))
# my_stop_words.extend('myword1 myword2 myword3'.split())
print(my_stopwords)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def my_preprocessor(doc):
    """
    expect: a string
    modify: 1. remove everything except alphabet charactors
            2. remove stopwords from customized stopwords list
    return: a string in lowercase
    """
    import re
    doc_clean = re.sub(r'[^\w]+|[\d,]+', ' ', doc)
    doc_list = doc_clean.split()
    doc_clean_list = [wd.lower() for wd in doc_list if wd.lower() not in my_stopwords ]
    return ' '.join(doc_clean_list)

# tokenize the doc and 'lemmatize' its tokens
def my_tokenizer(doc):
    """
    expect: a string
    modify: tokenize and lemmatize the string
    return: a list of tokens after lemmatization
    """
    import spacy
    # create a spaCy lemmatizer
    spacy.load('en_core_web_sm')
    lemmatizer = spacy.lang.en.English()
    lemmatizer.max_length = 3500000
    tokens = lemmatizer(doc)
    return ([token.lemma_ for token in tokens])

vect = CountVectorizer(max_df=0.95,min_df=5,
                       preprocessor=my_preprocessor, tokenizer=my_tokenizer)
vect.fit_transform(df.text)
vect.get_feature_names()

len(vect.get_feature_names())


# #### token_pattern = r'\b[a-zA-Z]{3,}\b'  and stemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# pattern = r'\b[^\d\W-_]+\b'
pattern = r'\b[a-zA-Z]{3,}\b'  ## raw string with word boundries, word at least 3 letter
vect = CountVectorizer(max_df=0.95,min_df=5, stop_words='english', token_pattern= pattern, tokenizer=my_tokenizer_stemmer)
vect.fit_transform(df.text)
vect.get_feature_names()

len(vect.get_feature_names())



# ### LDA Model
text_dtm = vect.transform(df.text)

from sklearn.decomposition import LatentDirichletAllocation
num_topics = 10

lda = LatentDirichletAllocation(n_components=num_topics, max_iter=100, learning_method='online')
lda_model = lda.fit_transform(text_dtm)

print("Size: ", lda_model.shape)  # (NO_DOCUMENTS, NO_TOPICS)
print("Component Size: ", lda.components_.shape)  # (WORDS, NO_TOPICS)


def topic_dist(model, doc=0, num_topic = 10):
    """
    expect: an object of lda model after fit.transform, an instance of vectorizer, and number of top words
    modify: 1. get the W factor (weights relative to each of the k topics)
               Each row corresponds to a diﬀerent document, and each column corresponds to a topic
            2. loop through each of document to get the indices of words in descending on log probabilities
    return: a list of top words for each of topics
    """
    import matplotlib.pyplot as plt
    topic_prob = list(model[doc]*100) # subset from W factor
    plt.figure(figsize=(12,8))
    plt.bar(x=[f'Topic {i+1}' for i in range(num_topic)], 
            height= topic_prob
           )
    file = df.loc[doc,['filename']].values[0]
    plt.title(f'Document #{file}')
    plt.xticks(range(num_topic))
    plt.ylabel('Percentage %')
    plt.show()

topic_dist(lda_model, doc=5)


def display_topics(model, vect, no_top_words=20):
    """
    expect: an instance of lda model from sklearn, an instance of vectorizer, and number of top words
    modify: 1. get the H factor (weights relative to each of the k topics)
               Each row corresponds to a topic, and each column corresponds to a unique term in the corpus vocabulary
            2. loop through each of document to get the indices of words in descending on log probabilities
    return: a list of top words for each of topics
    """
    feature_names = vect.get_feature_names()
    vector_H = model.components_
    for topic_idx, topic in enumerate(vector_H):
        print(f"\nTopic: {topic_idx}")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    # argsort() 
    #  It returns an array of indices of the same shape as that index data along the given axis in sorted order.
    # ex: array([900, 734, 171, 996, 303, 540, 321, 845, 108, 508], dtype=int64)

#### call the function above with model and CountV
display_topics(lda, vect)


# implement a print function 
# REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    """
    expect: an instance of lda model from sklearn, an instance of vectorizer, and number of top words
    modify: 1. get the H factor (weights relative to each of the k topics)
            Each row corresponds to a topic, and each column corresponds to a unique term in the corpus vocabulary
            2. loop through each of document to get the indices of words in descending on log probabilities
    return: a list of tuples that contains top words with log probabilities for each of topics
    """
    for idx, topic in enumerate(model.components_):
        print("\nTopic:  ", idx+1)
        feature_names = vect.get_feature_names()
        print([(feature_names[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])
    ## gets top n elements in decreasing order
    # argsort() 
    #  It returns an array of indices of the same shape as that index data along the given axis in sorted order.
    # ex: array([900, 734, 171, 996, 303, 540, 321, 845, 108, 508], dtype=int64)

#### call the function above with model and CountV
print_topics(lda, vect)


import matplotlib.pyplot as plt
import numpy as np

# get an object of H factor (weights relative to each of the k topics)
# ex: (10, 15732) --> 10 topics by 15732 words
word_topic = np.array(lda.components_)
# transpose the object of H factor (weights relative to each of the k topics) as topic on columns and words on rows
# ex: (15732, 10) --> 15732 words by 10 topics 
word_topic = word_topic.transpose()

num_top_words = 15

vocab_array = np.asarray(vect.get_feature_names())
plt.figure(figsize=(18,8))
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # subplot(row, column, plot_id_number ); plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t+1))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order for each of topic
    top_words_idx_limit = top_words_idx[:num_top_words] # narrow down the specific numbers of indices
    top_words = vocab_array[top_words_idx_limit]
#     top_words_shares = word_topic[top_words_idx, t]

    # iterate the words to show on the plot
    for i, word in enumerate(top_words):
        plt.text(0.1, num_top_words-i-0.3, word, fontsize=14)
plt.tight_layout()
plt.show()



####################################################
##
## INTERACTIVE VISUALIZATION
##
####################################################
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
# pip install pyLDAvis
pyLDAvis.enable_notebook() ## not using notebook
dtm = np.matrix(text_dtm.toarray()) 
panel = LDAvis.prepare(lda, dtm, vect, mds='tsne')
pyLDAvis.show(panel)


# https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af  
# https://gist.github.com/JoseHJBlanco/4a22629c4bd925bb2e1bdfb4fe627039  
# https://www.youtube.com/watch?v=SF50IK5XgKA



