# Core Pkgs
import streamlit as st
import string
import re

# NLP Packages
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib

# Vizualization Package 
import matplotlib.pyplot as plt

# In-built Class
from get_sentiments import *
from get_emotions   import *

def identify_problem_areas(pre_processed_text):
    
    ''' Function to identify problem areas '''
    problem_area_list = []
        
    file_car_parts = open("./static/car_parts.txt", "r")
    car_parts_list = file_car_parts.read().split()
    
    for word in pre_processed_text:
        if word in car_parts_list:
            problem_area_list.append(word)
        
    if len(problem_area_list) > 0 :
        return list(set(problem_area_list))
    else:
        return 'Could not find any matching words'
    
def perform_data_pre_processing(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    
    word_list_clean = []
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    
    text = lemmatizer.lemmatize(text) # Lemmatize (Root Words)
    
    text_list = text.split()
    
    for word in text_list:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
                word_list_clean.append(word)
            
    return word_list_clean

def read_neg_pos_word_dict(polarity_info, pre_processed_text):
    
    ''' This file read the negative & postive word from files '''
    
    neg_feedback_words = []
    pos_feedback_words = []
        
    if polarity_info == 'Negative':
        
        file_neg_lex = open("./static/negative-words.txt", "r")
        neg_lex_list = file_neg_lex.read().split()
        
        for neg_word in pre_processed_text:
            if neg_word in neg_lex_list:
                neg_feedback_words.append(neg_word)
        
        if len(neg_feedback_words) > 0 :
            return list(set(neg_feedback_words))
        else:
            return 'Could not find any matching words'
    
    else:
        
        file_pos_lex = open("./static/positive-words.txt", "r")
        pos_lex_list = file_pos_lex.read().split()

        for pos_word in pre_processed_text :
            if pos_word in pos_lex_list:
                pos_feedback_words.append(pos_word)
        
        if len(pos_feedback_words) > 0 :
            return list(set(pos_feedback_words))
        else:
            return 'Could not find any matching words'
    

def extract_key_values_from_dict(input_dict):
    
    '''  Function to extract keys & values from dictionary for plotting '''
    labels = []
    sizes  = []

    for key, value in input_dict.items():
        if value > 0:
            labels.append(key)
            sizes.append(value)
        
    return labels, sizes
    
    
def sentiment_pie_chart(sizes, labels):
    
    ''' Function For Pie Chart '''
    
    fig1, ax1   = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    
    return fig1
        
def load_sentiment_analysis_ui():
    
    # Real Time Search Box
    with st.form(key='emotion_clf_form'):
        st.header('Analyze Your Sentence')
        raw_text = st.text_area("")
        submit_text = st.form_submit_button(label='Submit')
    
    if submit_text:
        
        col1,col2  = st.columns(2)
        
        # Get Sentiments
        sentiment_polarity = analyzeSentiments(raw_text)       
        
        # Get Emotions
        emotions_info  = Emotions(raw_text)
        
        # Clean Text Data for Core words
        pre_processed_text = perform_data_pre_processing(raw_text)
        
        with col1:
            
            # Polarity Extractions
            polarity_info = sentiment_polarity.get_sentiment_polarity()
            polarity_df = sentiment_polarity.get_sentiment_data()
            
            if  polarity_info == 'Positive':           
                st.success(f'Sentiment - {polarity_info}')
                word_list = read_neg_pos_word_dict(polarity_info, pre_processed_text)
                
            elif polarity_info == 'Negative':
                st.error(f'Sentiment - {polarity_info}')
                word_list = read_neg_pos_word_dict(polarity_info, pre_processed_text)
                
            else:
                st.info(f'Sentiment - {polarity_info}')
                word_list = read_neg_pos_word_dict(polarity_info, pre_processed_text)
                
            st.pyplot(sentiment_pie_chart(list(polarity_df['Score']), list(polarity_df['Sentiment'].unique())))
            
            st.info('Customer Feedback - ')
            st.write(word_list)
            
        with col2 :
            
            st.info('Emotions')
            labels, sizes = extract_key_values_from_dict(emotions_info.extract_emotions())
            st.pyplot(sentiment_pie_chart(sizes, labels))
            
            st.info(f'Problem Area')
            st.write(identify_problem_areas(pre_processed_text))
            
        
if __name__ == "__main__":
    load_sentiment_analysis_ui()