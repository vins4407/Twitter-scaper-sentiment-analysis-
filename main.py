import csv
import tweepy
import ssl
from fastapi import FastAPI
from typing import List   
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

ssl._create_default_https_context = ssl._create_unverified_context

# Replace the following with your own Twitter API credentials
consumer_key = "your-consumer-key"
consumer_secret = "your-consumer-secret"
access_token = "access-token" 
access_token_secret = "access-token-secret"


app = FastAPI()
Hashtag_tweets = []

#function for getting tweets based on hashtag
def get_Hashtag_tweets(hashtag):
  
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        # Search for tweets with the specified hashtag
        tweets = tweepy.Cursor(api.search_tweets, q=f"#{hashtag}", result_type='popular').items(25)

        for tweet in tweets:
            # Extract relevant information for abstraction
            tweet_text = tweet.text
            Hashtag_tweets.append(tweet_text)
        
        
        return get_sentiments(Hashtag_tweets)

    except tweepy.TweepError as e:
        return {"error": f"Error occurred while retrieving tweets: {e}"}

#Fast-api for serving the hashtag data
@app.get("/get_hashtag_tweets/")
def get_hashtag_tweets_endpoint(hashtag: str):
    return get_Hashtag_tweets(hashtag),Hashtag_tweets



#function for getting the tweets and replies under the tweet
def get_tweet_replies(tweet_url):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    
    # Extract tweet ID and Twitter username from the URL
    tweet_id = tweet_url.split("/")[-1]
    username = tweet_url.split("/")[3]

    try:
        # Retrieve the tweet using its ID
        tweet = api.get_status(tweet_id, tweet_mode='extended')

        replies = []
        for tweet in tweepy.Cursor(api.search_tweets, q='to:' + username, result_type='recent', timeout=999999).items(1000):
            if hasattr(tweet, 'in_reply_to_status_id_str') and tweet.in_reply_to_status_id_str == tweet_id:
                # Remove the usernames starting with '@' from the reply
                reply_text = ' '.join(word for word in tweet.text.split() if not word.startswith('@'))
                replies.append(reply_text)
        
        filtered_data = [text for text in replies if text.strip() != '']
        # Now 'replies' contains a list of reply messages without the usernames
        print(filtered_data)
        return get_sentiments(filtered_data),filtered_data

    except tweepy.TweepError as e:
        return {"error": f"Error occurred while retrieving tweets: {e}"}
         
#api route for delivering the request response
@app.get("/tweet_replies/")
def tweet_replies(tweet_url: str):
    return get_tweet_replies(tweet_url)


#function that uses Bert model for sentiment analysis
def get_sentiments(data):
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    # Given texts
    texts = [line.replace('\n', '') for line in data]
    print(texts)
   
        
    # List to store sentiment labels
    sentiments = []

    # Perform sentiment analysis for each text and store the results
    for text in texts:
        # Tokenize the text
        tokens = tokenizer.encode(text, return_tensors='pt')
        
        # Get model predictions (logits)
        result = model(tokens)
        logits = result.logits
        
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1).tolist()[0]

        # Determine the sentiment label
        sentiment_label = ["negative", "neutral", "positive"]

        # Check if probabilities list is valid
        if all(isinstance(p, float) for p in probabilities):
            max_prob_index = probabilities.index(max(probabilities))
            if max_prob_index < len(sentiment_label):
                sentiment_label = sentiment_label[max_prob_index]
            else:
                sentiment_label = "unknown"

        # Append sentiment label to the list
        sentiments.append(sentiment_label)

    # Count the number of positive, negative, and neutral sentiments
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    neutral_count = sentiments.count("unknown")

    # Calculate the total number of texts
    total_texts = len(sentiments)

    print(sentiments)
    # Check if there are texts before calculating the percentages
    if total_texts > 0:
        # Calculate the percentage of each sentiment
        positive_percentage = (positive_count / total_texts) * 100
        negative_percentage = (negative_count / total_texts) * 100
        neutral_percentage = (neutral_count / total_texts) * 100

        # Determine the sentiment with the maximum count
        max_sentiment_count = max(positive_count, negative_count, neutral_count)

        # Return the sentiment with the maximum count as a string
        if max_sentiment_count == positive_count:
            max_sentiment = f"Positive Sentiment: {positive_count}  ({positive_percentage:.2f}%)"
        elif max_sentiment_count == negative_count:
            max_sentiment = f"Negative Sentiment: {negative_count}  ({negative_percentage:.2f}%)"
        else:
            max_sentiment = f"Neutral Sentiment: {neutral_count}  ({neutral_percentage:.2f}%)"

        return max_sentiment
    else:
        return "No texts to process."














