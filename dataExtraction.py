import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd
import re
from nltk.corpus import stopwords
def cleanText(rawText):
    letters_only = re.sub("[^a-z,A-Z]"," ",str(rawText))
    lower_case = letters_only.lower()
    words = lower_case.split()
    stops = set(stopwords.words("english"))
    useable_words = [w for w in words if not w in stops]
    return " ".join(useable_words)
ckey = '**********************'
csecret = '*****************************'
atoken = '****************************************'
asecret = '***********************************'

#Maximum Count using twitter api in one round is 100 only.

auth = OAuthHandler(ckey , csecret)
auth.set_access_token(atoken, asecret)
api = tweepy.API(auth)
public_tweets = api.search('#Trump', count=100)

#This is a document which contains all the tweets.
documents = []
for tweet in public_tweets:
    documents.append(tweet.text.encode('utf-8'))

# Now applying concept of removal of stopwords on these tweets.
documents1 = []
for tweet in documents:
    stre = cleanText(str(tweet))
    documents1.append(stre)

print "Tweets after removal of stopwards"
for tweets in documents1:
    print tweets

