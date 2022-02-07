# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("./model/emotion_classifier_model.pkl","rb"))

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Main Application
def main():
    st.title("Emotion Classifier App")
    
    st.subheader("Home-Emotion In Text")
    
    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1,col2  = st.columns(2)
        
        # Apply Fxn Here
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            # st.success('Original Text')
            # st.write(raw_text)
            
            st.success('Prediction')
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(emoji_icon  + ' : ' +  prediction)
            st.write(f'Confidence_Score = {np.round(np.max(probability), 2)}')
            
        with col2 :
            st.success('Prediction Probability')
            
            emotion = pd.DataFrame(
                    {'Emotions' : pipe_lr.classes_, 
                    'Probability': probability[0]
                    }
                    )
            emotion = emotion.sort_values(by='Probability', ascending =False)
            # st.dataframe(emotion)
            
            fig = alt.Chart(emotion).mark_bar().encode(x='Emotions',y='Probability',color='Emotions')
            st.altair_chart(fig,use_container_width=True)


if __name__ == '__main__':
	main()