#!/usr/bin/env python
# coding: utf-8

# # HW3 Corpus Analysis

# Dataset 1: (Topic: Restaurants and restaurant reviews)
# https://drive.google.com/file/d/11H6AbWxKsPLY3yt__OrmK0rjjYShKhig/view?usp=sharing
# 
# Dataset 2: (Topic: Movie reviews)
# https://drive.google.com/file/d/17nGHPsk4RXfvRoq-ndizTzc0_0PkzAqm/view?usp=sharing
# 
# Both of these datasets are csv files. However, Dataset 1 is easier to clean and prepare. Dataset 2 is more of a challenge.
# 
# Your goal for both is to use Python and CountVectorizer and to prepare and format the datasets into dataframe. The first column in each dataframe should be called “LABEL” and will be the label of the data for that row. 
# 
# *** Because I am giving you two datasets to prepare, I am WAIVING the FORMAT – this time only. You DO NOT need to have an Intro or a Conclusions. 
# 
# In Analysis you will talk about the two datasets, how they are similar, how they are different, and the challenges you faced in preparing and formatting each into labeled data frames. 
# 
# In Results you will include images of each of the dataframes (partial because they are huge) for each data set. 
# 
# Show the BEFORE (the raw data) and the AFTER (the clean and labeled dataframe). Do this for each dataset. 
# 
# Keep in mind that because the dataset and dataframe are BIG – you will only need to create images that show portions of each. 
# 



import pandas as pd
import re


pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 1000)


# ## Dataset 1


restaurant = pd.read_csv('RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE.csv')
print('The sample dataset has {} document and {} columns'.format(restaurant.shape[0], restaurant.shape[1]))
restaurant.head()


restaurant.isna().sum().sum()


# Summary:
#    * 9 instances
#    * labels are located at fisrt column for each row
#    * text data contain newline
#    * back slash and apostrophe have to be removed
#    
# 
# #### Strategies:
# 1. Fill NAs with spaces
# 2. forloop each rows and use string method to join each fo reviews
# 3. strip() method to get rid of leading and trailing spaces
# 4. rebuild a dataframe


# fill na with a space
restaurant = restaurant.fillna(' ')


def df_string_to_list(dataframe):
    '''
    expect: a dataframe with seperated strings in different columns 
    modify: fill the NAs with spaces, and combine the strings from each column for each row as a document
    return: a list of documents
    '''
    dataframe = dataframe.fillna(' ')
    # create two temp lists
    # second temp_list is for storing a list of string list for each document
    # final_list is for storing a list of strings for each document
    second_temp_list=[]
    final_list=[]
    for i, r in dataframe.iterrows():
        # create a temp list to store a list of strings for a document
        temp_list=[]
        for s in r:
            # append strings as a list of strings for a document
            temp_list.append(s)
        # append the string list for each documents
        second_temp_list.append(temp_list)
    # join the string list and remove trailing and leading spaces for each document    
    final_list = [' '.join(second_temp_list[i]).strip() for i in range(len(second_temp_list))]
    return final_list


review = df_string_to_list(restaurant.iloc[:,1:])


# create a new dataframe
restaurant_df = pd.DataFrame({'label': restaurant.sentiment, 'review':review})
# it is still messy in the strings
restaurant_df.head()



# define functions for cleaning strings in dataframe
def clean_str(string):
    '''
    expect: an unclean string. ex: "'Mike\\'s Pizza High Point  NY Service was very slow and the quality was low.'"
    modify: remove new line and back slash; remove leading and trailing back slash and apostrophe
    return: a string
    '''
    import re
    temp = re.sub(r'\\n|\\','', string) # remove new line and back slash
    temp2 = temp.strip("'")       # remove leading and trailing apostrophe
    result = temp2.strip("\\")       # remove leading and trailing back slash
    return result


# clean the unwanted strings
restaurant_df['review'] = restaurant_df.review.map(clean_str)
restaurant_df['label']= restaurant_df.label.map({'n':'neg', 'p':'pos'})
restaurant_df.head()


restaurant_df.to_csv('restaurant_clean.csv')




# ## Dataset 2


pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 60)


movie = pd.read_csv('MovieDataSAMPLE_labeledVERYSMALL.csv')
print('The sample dataset has {} document and {} columns'.format(movie.shape[0], movie.shape[1]))
movie.head()


movie.isna().sum().sum()


# Summary:
#    * 5 instances
#    * labels are located at last column for each row
#    * text data contain newline
#    * back slash and apostrophe have to be removed
#     
# 
# #### Strategies:
# 1. a nested for loops to iterate each row
# 2. store review and label to lists
# 3. create pandas dataframe with two lists as dictionary format


review_list =df_string_to_list(movie)
label=[review_list[i][-3:] for i in range(len(review_list))]
review=[review_list[i][:-3].strip() for i in range(len(review_list))]


pd.set_option('display.max_colwidth', 1000)
# create a new dataframe
movie_df = pd.DataFrame({'label': label, 'review':review})
# it is still messy in the strings
movie_df.head()


movie_df.review = movie_df.review.map(clean_str)
movie_df.head()


movie_df.to_csv('movie_clean.csv')

