# Core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib

# In-built Class
from get_sentiments import *

# Vizualization Package 
import matplotlib.pyplot as plt

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
        
        with col1:
            # Polarity Extractions
            polarity_info = sentiment_polarity.get_sentiment_polarity()
            polarity_df = sentiment_polarity.get_sentiment_data()
            
            # Pie Chart Configurations
            labels = list(polarity_df['Sentiment'].unique())
            sizes = list(polarity_df['Score'])
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            #
            st.info(polarity_info)
            # st.write(polarity_df.set_index('Sentiment'))
            st.pyplot(fig1)
            
        with col2 :
            st.success('Emotions')
            
        
if __name__ == "__main__":
    load_sentiment_analysis_ui()