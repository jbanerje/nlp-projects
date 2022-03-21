# Rule-based sentiment analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline


# Data Manipulation Library
import pandas as pd

def perform_transformer_sent_analysis(usr_text):
    
    ''' Sentiment Analysis Using Transformer '''
    # pip install transformers
    sentiment_pipeline = pipeline("sentiment-analysis")
    print(sentiment_pipeline(usr_text))
    
def perform_flair_sent_analysis(usr_text):
    
    ''' Sentiment Analysis Using Flair '''
    # pip install flair
    classifier = TextClassifier.load('en-sentiment')
    sentence = Sentence(usr_text)
    classifier.predict(sentence)
    
    # print sentence with predicted labels
    print('Sentence above is: ', sentence.labels)

    return

def perform_vader_sent_analysis(usr_text):
    
    ''' Sentiment Analysis Using Vader Sentiment Analysis '''
    
    key_mapper = {'neu':'Neutral', 'pos':'Positive', 'neg':'Negative'}
    
    score_dict = SentimentIntensityAnalyzer().polarity_scores(usr_text)
    
    # Polarity Dataframe
    polarity_df = pd.DataFrame([score_dict])[['pos', 'neg', 'neu']].T.reset_index().rename(columns={'index':'Sentiment', 0:'Score'}).sort_values(by='Score', ascending=False)
    
    polarity_df['Sentiment'] = polarity_df['Sentiment'].map(key_mapper)
    polarity_df = polarity_df[polarity_df.Score > 0]

    print(polarity_df)
    
    return
    
def perform_text_blob_sent_analysis(usr_text):
    
    ''' Sentiment Analysis Using Textblob '''
    
    testimonial = TextBlob("The food was great!")
    print(testimonial.sentiment)
    
    return
 
if __name__ == '__main__':
    
    ''' Main Function '''
    usr_text = 'The food was great!'
    # perform_text_blob_sent_analysis(usr_text)
    # perform_vader_sent_analysis(usr_text)
    # perform_flair_sent_analysis(usr_text)
    perform_transformer_sent_analysis(usr_text)