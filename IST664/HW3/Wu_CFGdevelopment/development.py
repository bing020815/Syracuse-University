'''
This small development program can be run to parse the sentences for Homework 3.
You may either run this as a stand-alone python program or copy it to jupyter notebook.
In either case, it must be in the same directory as the sentences and the grammar.
'''

import nltk

# read the sentences from the file sentences.txt
sentfile = open('sentences.txt', 'r')
# make a list of sentences, separating the tokens by white space.
sentence_list = []
for line in sentfile:
    sentence_list.append(line.split())

# read the grammar file - the nltk data function load will not reload
#    the file unless you set the cache to be False
camg = nltk.data.load('file:camelot_grammar.cfg', cache=False)

# create a recursive descent parser
cam_parser = nltk.RecursiveDescentParser(camg)

# for each sentence print it and its parse trees
# if the grammar cannot parse a sentence, sometimes it gives an error and
#    sometimes it just goes on to the next sentence with no parse tree
for sent in sentence_list:
    print(sent)
    for tree in cam_parser.parse(sent):
        print (tree)
        print()
