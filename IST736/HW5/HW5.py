#!/usr/bin/env python
# coding: utf-8

# NOTE1: You do not have to use the docs or data from HW1. You can find, use, or create any data you wish for this assignment. 
# 
# and use AMT platform to request five sentiment labels per document. Note that you can set up your tasks in different ways, such as hiring five workers, each annotating all comments, or hiring 10 workers, each annotating half of the comments, etc.
# 
# NOTE2: Class – the goal here is to realize that sometimes you need labeled data, but do not have it. This happens in real life very often. I suggest that you find, use, or create 6 small doc – __three (3) positive__ and __three (3) negative__ – such as movie or restaurant reviews. Make 4 of them really easy. For example, an easy one that is positive would have words like good, like, love, great, excellent, etc. or for negative, words like bad, poor, dislike, etc. Then, make two of the six ambiguous  - such as using sarcasm, etc. These are just ideas. You may use whatever you wish. The goal is ot create something that makes sense to you, so that you understand the results. 
# 
# Next, you will practice using something new. Amazon Mechanical Turk (AMT). Learn about it. Google it. Then use it on your 6 (or so) docs. Did the “Turkers” labels correctly? What did they get wrong? How long did it take? How much did it cost? Etc. 
# 
# After obtaining the labels, calculate pair-wise Kappa values among the AMT workers, and then calculate the average Kappa value as the overall agreement among the AMT workers. Also calculate the Kappa agreement between each AMT worker's annotations and your manual annotations. Based on these calculations, discuss the AMT workers' annotation reliability.
# 
# NOTE 3: Use YouTube to learn how to calculate Kappa. The math is easy and so this is a great opportunity to practice taking the lead and learning something new without a Guide. 
# 
# Describe in your report:
# 
# NOTE 4: Please use the Assignment Format. Remember – the topic is whatever the dataset is about. The Analysis is the AMT and labeling part. The Results are what happened. The conclusions might include whether a restaurant owner (if you data is about restaurants for example0 can use this method to better understand the sentiment of 1 million reviews, etc. 
# 
# Also include the following in the appropriate location in the format assignment….
# 
# a.	The experiment design: How many turkers do you aim to hire? What is the workload and payment for each turker? What is your requirement for the turkers (language proficiency, geographical location, past work performance, etc.)? Please also explain why you think this is the best choice for your experiment design to obtain the best-quality data in the most efficient way, e.g., your spam-control strategy.  
# 
# b.	The experiment outcome: How long did it take to obtain all labels? How much did you pay in total? Did any unexpected events occur during the process? Did you find any spammers? If yes, how did you find out and remove spam data? What is the average Kappa agreement among the workers? What are their levels of agreement with your ground truth? Do all AMT workers share similar marginal distributions?  
# 
# c.	Conclusion: Do you think AMT is a viable approach for obtaining training labels? What lessons did you learn in this experiment?  
# 


import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_colwidth', 1000)

yelp = pd.read_csv('text.csv', index_col='doc')


yelp.loc[:,['stars', 'text', 'label']]

sentiment_map = {'n':0, 'p':1}
AMT = yelp.loc[:,'label':'R'].copy()
AMT


doc = ['doc1','doc2','doc3','doc4','doc5', 'doc6']
tucker = [5,5,5,5,5,5]
vote = [(2/5), (3/5), (2/5), 1, 1, (2/5)]
match = [0,0,0,1,1,0]
tucker_df = pd.DataFrame({'doc':doc, 'tucker':tucker, 'vote':vote})

tucker_df.plot(kind='barh', x='doc', y='vote')
plt.title('Votes from Tuckers')

yelp.columns


# #Truth P vs Truker Q
from sklearn.metrics import confusion_matrix
import seaborn as sns

cfm = confusion_matrix(['n','n','p','n'],['p','p','p','p']) # P and Q
cmap = sns.diverging_palette(258, 0, n=2, as_cmap=True)
hm = sns.heatmap(cfm,annot=True, cmap=cmap)
bottom, top = hm.get_ylim()
hm.set(xticklabels=['n', 'p'])
hm.set(yticklabels=['n', 'p'])
hm.set(ylabel='Annotator P', xlabel='Annotator Q')
hm.set_ylim(bottom + 0.5, top - 0.5)



# # Ground Truth vs Trukers
GroundTruth = ['p','p','p','p','p',
               'n','n','n','n','n',
               'p','p','p','p','p',
               'n','n','n','n','n',
               'n','n','n','n','n',
               'p','p','p','p','p']
TuckerLabel = ['p','p','p','p','p',
               'p','n','p','n','n',
               'p','p','p','p','p',
               'p','n','n','p','n',
               'n','n','p','p','n',
               'n','n','p','p','p']

cfm = confusion_matrix(GroundTruth,TuckerLabel)
cmap = sns.diverging_palette(258, 0, n=2, as_cmap=True)
hm = sns.heatmap(cfm,annot=True, cmap=cmap)
bottom, top = hm.get_ylim()
hm.set(xticklabels=['n', 'p'])
hm.set(yticklabels=['n', 'p'])
hm.set(ylabel='Ground Truth', xlabel='Tucker Label')
hm.set_ylim(bottom + 0.5, top - 0.5)

