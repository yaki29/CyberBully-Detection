import tweepy
import csv

consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth,wait_on_rate_limit_notify=True, wait_on_rate_limit=True)


csvFile = open('galliya.csv', 'a')
bdfile = open('bdfile.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

csvWriter.writerow(["Date", "Tweet"])

count = 1

for query in bdfile:
	count = 1
	for tweet in tweepy.Cursor(api.search,q=query,count=100,\
	                           lang="en",\
	                           
	                           untill="2017-05-10").items():
	    print(tweet.created_at, tweet.text)
	    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
	    print(count)
	    count = count + 1
	    if(count > 50):
	    	break

print("Ended")
