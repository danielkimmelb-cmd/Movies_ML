"""
Building a model to conduct Sentiment analysis of movie and 
show description to analyze correlation between sentiment and
audience reviews. Extension : Words in title vs Popularity  
(finding the ideal movie title that would lead to virality and popularity)	
"""

import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# Preprocessing
df = pd.read_csv('Datasets/titles.csv')

df['seasons'] = df['seasons'].apply(lambda x: 0 if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x)

df.dropna(inplace=True)

def remove_special_characters(text):
    text = text.casefold()
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

df['description'] = df['description'].apply(remove_special_characters)
print(stopwords.words('english'))
# Tokenization and Remove Stop words
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


df['description'] = df['description'].apply(tokenize_and_remove_stopwords)
df['title'] = df['title'].apply(tokenize_and_remove_stopwords)


# Sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    """ 
    compound score from -1 to +1,
    meaning most negative and most positive
    """
    sentiment = sia.polarity_scores(text)
    # print(sentiment)
    return sentiment['compound']

df['sentiment_score_description'] = df['description'].apply(get_sentiment_scores)
df['sentiment_score_titles'] = df['title'].apply(get_sentiment_scores)

df.to_csv('everything.csv', index=False)
