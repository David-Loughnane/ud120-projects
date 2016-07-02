#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter - limit no. emails processed
#temp_counter = 0
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        #if temp_counter < 100:
        path = os.path.join('..', path[:-1])
        print(path)
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        stemmed_email = parseOutText(email)
        ### use str.replace() to remove any instances of the words
        words_to_replace = ["sara", "shackleton", "chris", "germani"]
        for word in words_to_replace:
            if (word in stemmed_email):
                stemmed_email = stemmed_email.replace(word, "")
        #stemmed_removed_email = " ".join([word for word in stemmed_email.split() if word not in words_to_replace])
        ### append the text to word_data

        word_data.append(stemmed_email)
        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "sara":
            from_data.append(0)
        else:
            from_data.append(1)

        email.close()


print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors.pkl", "wb") )


### in Part 4, do TfIdf vectorization here

#import list of stopwords from nltk corpus
#sw = stopwords.words("english")
#creates a vector of tfidf values with stopwords removed
tfidfVec = TfidfVectorizer(stop_words='english')

wordVec = tfidfVec.fit_transform(word_data)
print("Number of words: ", len(tfidfVec.get_feature_names()))
print("Word 34597: ", tfidfVec.get_feature_names()[34597])