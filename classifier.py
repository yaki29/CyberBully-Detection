import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def cleanText(rawText):
    letters_only = re.sub("[^a-z,A-Z]"," ",str(rawText))
    lower_case = letters_only.lower()
    words = lower_case.split()
    stops = set(stopwords.words("english"))
    useable_words = [w for w in words if not w in stops]
    return " ".join(useable_words)

train = pd.read_csv("data.csv", header=0, delimiter=",", quoting=3)

trainingSetSize = train["text_message"].size
print "Cleaning and preparing text_message training set for further processing.........."
clean_train_review=[]
for i in xrange(0,trainingSetSize):
    if i%1000==0:
        print "text_message %d of %d" %(i+1,trainingSetSize)
    clean_train_review.append(cleanText(train["text_message"][i]))

print "Training set reviews are cleaned and ready for further processing\n"
print "Creating Bag of Words .................\n"
vectorizer=CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 50000)

train_data_features = vectorizer.fit_transform(clean_train_review)
#train_data_features=TfidfTransformer(use_idf="false").fit_transform(train_data_features)
train_data_features=train_data_features.toarray()
print train_data_features.shape
filename1 = 'finalized_vector.sav'

pickle.dump(vectorizer, open(filename1, 'wb'))
classifier = LogisticRegression()

classifier = classifier.fit(train_data_features,train["label_bullying"])
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
# print clean_train_review.shape,"  ::: "
# print train["label_bullying"].shape
print classifier.score(train_data_features,train["label_bullying"])
ans = 0
truePositive = 0;
trueNegative = 0;
falsePositive = 0;
falseNegative = 0;
totalPositive = 0;
totalNegative = 0;
# print train["label_bullying"]
for i in xrange(0,trainingSetSize):
    if i%1000==0:
        print "text_message %d of %d" %(i+1,trainingSetSize)
    # print
    test = []
    test.append(clean_train_review[i])
    # print clean_train_review[i]," :::: ",test
    test = vectorizer.transform(test)
    test = test.toarray()
    if(test.shape[0] is 0):
        continue
    # result = classifier.predict(train_data_features[i].reshape(1,-1))
    result = classifier.predict(test)
    p = train["label_bullying"][i]
    # print i," ",result," ",p
    if(int(result[0]) is int(p)):
        ans = ans + 1
    if(int(p) is 1 and int(result[0]) is 1):
        truePositive = truePositive + 1
    elif(int(p) is 0 and int(result[0]) is 0):
        trueNegative = trueNegative + 1
    elif(int(p) is 1 and int(result[0]) is 0):
        falseNegative = falseNegative + 1
    elif(int(p) is 0 and int(result[0]) is 1):
        falsePositive = falsePositive + 1
    if(int(p) is 1):
        totalPositive = totalPositive + 1;
    if(int(p) is 0):
        totalNegative = totalNegative + 1;
    # print "ans is ",ans

print ans
print "accuracy is ",((1.0*ans)/trainingSetSize)*100
print "truePositive " , (truePositive)
print "trueNegative " , (trueNegative)
print "falsePositive ", (falsePositive)
print "falseNegative ", (falseNegative)
print "totalPositive ", (totalPositive)
print "totalNegative ", (totalNegative)
print "truePositive rate " , (( (1.0*truePositive)/totalPositive)*100.0)
print "trueNegative rate" , (((1.0*trueNegative)/totalNegative)*100.0)
print "falsePositive rate" , (( (1.0*falsePositive)/totalPositive)*100.0)
print "falseNegative rate" , (( (1.0*falseNegative)/totalNegative)*100.0)