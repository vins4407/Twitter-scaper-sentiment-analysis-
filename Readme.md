# Sentiment Analysis Twitter API

![Twitter API Logo](https://example.com/twitter-api-logo.png)

## Overview

The Sentiment Analysis Twitter API is a Python-based tool that allows users to perform sentiment analysis on Twitter data. With this API, you can extract tweets based on hashtags and analyze their sentiments. Additionally, you can analyze replies under a specific tweet to gauge public responses.

This project utilizes FastAPI for building the API, Tweepy for accessing Twitter data, Transformers library for sentiment analysis using the BERT model, and TextBlob for NLP tasks.

## Features

- **Hashtag Tweets Sentiment Analysis:** Fetch popular tweets associated with a hashtag and analyze their sentiments, classifying them as positive, negative, or neutral.

- **Tweet Replies Sentiment Analysis:** Retrieve a tweet along with its replies and perform sentiment analysis on the replies, categorizing them into positive, negative, or neutral sentiments.

## How to Use

1. **Twitter API Credentials:** Obtain your Twitter API credentials from the [Twitter Developer Dashboard](https://developer.twitter.com/en/portal/projects-and-apps) and replace the placeholder credentials in the code.

2. **Install Dependencies:** Ensure you have the required Python dependencies installed. Run the following command to install the necessary packages:

```bash
pip install tweepy fastapi textblob transformers torch
```
## How to Use

 **Run the API:**
   Execute the following command in your terminal or command prompt:

   ```bash
    uvicorn main:app --reload 
  ```

This will start the FastAPI development server, and the API will be up and running.

    API Endpoints:

        GET /get_hashtag_tweets/:
        Provide a hashtag as a parameter to get popular tweets associated with it and their sentiment analysis results.

        GET /tweet_replies/:
        Supply a tweet URL as a parameter to fetch the tweet and its replies, then perform sentiment analysis on the replies.

    Interpreting Sentiment Analysis Results:
    The API uses the BERT model for sentiment analysis, which classifies texts as positive, negative, or neutral based on model predictions.

Once the API is running, you can use tools like curl, Postman, or any programming language's HTTP client library to make requests to the API endpoints and retrieve sentiment analysis results for tweets and replies.

For example, using curl, you can make a request to the API endpoint to get hashtag tweets sentiment analysis:

bash

curl -X 'GET' \
  'http://localhost:8000/get_hashtag_tweets/?hashtag=AI' \
  -H 'accept: application/json'

And the API will respond with the sentiment analysis results for the hashtag tweets.

Similarly, you can make a request to the API endpoint to get tweet replies sentiment analysis:

bash

curl -X 'GET' \
  'http://localhost:8000/tweet_replies/?tweet_url=https://twitter.com/username/status/1234567890' \
  -H 'accept: application/json'

Remember to replace http://localhost:8000 with the actual URL where your API is hosted.

Feel free to integrate this API into your projects, applications, or services to leverage Twitter sentiment analysis for better insights into public opinions and trends.