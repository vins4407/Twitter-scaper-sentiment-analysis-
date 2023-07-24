from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Given texts
texts = [
  "Hey I'm interested. portfolio:- https://t.co/DBMNdJw372",
  "https://t.co/enmwl6iRvw I ❤️ react, and building tool chains. Hope you like my portfolio!",
  "Hey Abhishek, I am Harsh Gautam from NIT-Bhopal. I will be happy to work with the team… https://t.co/o0PTTTwUnf",
  "Hey i feel interested for this role Here’s my resume https://t.co/PevVWk78Ri",
  "Hello sir, I am intrested, I have texted you with my profile details please do check it.",
  "Here is my portfolio: https://t.co/rn1yI7gIKe",
  "Pls check dm",
  "Heyy interested check your dm!",
  "Hey, I'm an app and a web dev 3rd yr student. Portfolio - https://t.co/ODwS7384fu. I can… https://t.co/DrsubzsArr",
  "https://t.co/YJgZcrhYml",
  "Interested",
  "I am interested and here's my portfolio https://t.co/5ju557ZSkd",
  "Didn't y'all withdraw PPO offers this cycle?? https://t.co/Kg1yPTSUhe",
  "It's not allowing to DM Sharing the portfolio here - https://t.co/En9CAxsome",
  "its time to get back to web2 job",
  "Intrested https://t.co/KI5uNX5H3d",
  "I've done a dm with my resume. please check it out! Thanks in advance.",
  "Hi, I'm a full stack developer and have experience in various frontend technologies like… https://t.co/7tkDvbF5kA",
  "Hey . I want to know more about the detail",
  "Portfolio- https://t.co/fF6NSB0P0J",
  "is there full time roles for freshers ?",
  "Interested. https://t.co/w1nOY8zUwI",
  "is literally the only person u would need",
  "check your dm please",
  "Interested",
  "Checkout https://t.co/ZO1tHMENJZ TAT 3 hours",
  "Hello sir please check your DM",
  "Hey I've sent you a dm with my resume. Kindly check it.",
  "Cc:",
  "cc:",
  "Hey I'm interested. I'd like to know more about the opportunity. Portfolio:- https://t.co/L2DFkcqQwM",
  "Interested Please check your dm",
  "check this. =)",
  "hi, only verified can dm you. therefore unable to dm",
  "Any opening for Android intern?Would love to apply"
]
    

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
        print(f"Probabilities: {probabilities}")
        max_prob_index = probabilities.index(max(probabilities))
        print(f"Max Probability Index: {max_prob_index}")
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
    print("this is total text: {total_texts}")
    positive_percentage = (positive_count / total_texts) * 100
    negative_percentage = (negative_count / total_texts) * 100
    neutral_percentage = (neutral_count / total_texts) * 100

    # Print the sentiment statistics
    print("Sentiment Analysis for the Entire Data:")
    print(f"Total Texts: {total_texts}")
    print(f"Positive Sentiment: {positive_count} texts ({positive_percentage:.2f}%)")
    print(f"Negative Sentiment: {negative_count} texts ({negative_percentage:.2f}%)")
    print(f"Neutral Sentiment: {neutral_count} texts ({neutral_percentage:.2f}%)")
else:
    print("No texts to process.")
