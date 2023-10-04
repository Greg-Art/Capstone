import streamlit as st
import plotly as plt 
import numpy as np 
import pandas as pd
import webbrowser
import requests
import json
from streamlit_lottie import st_lottie 

st.set_page_config(page_title= "Welcome Page", page_icon ="👋")


st.sidebar.success("Select The Page You Want to Explore: ")

st.title("Welcome to my Sentiment Analysis App")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# initializaing my session state 
if 'lottie_hello' not in st.session_state:
    st.session_state.lottie_hello = load_lottiefile("./lottie_animations/main.json")

# creating a funciton to upload the file while implementing session state
def handle_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        st.session_state.lottie_hello = load_lottiefile(uploaded_file.name)


# displaying the Lottie animation
st_lottie(st.session_state.lottie_hello, height=200)


st.markdown("""On this app, you will  be able to classify Movie Review sentiments with the Tiny-Bert model
The objective of this challenge is to develop a machine learning model to assess if a movie review is positive or negative.""")

st.subheader("""Variable Definition:""")

st.write("""

    **Review File**: Unique identifier of the review

**Content**: Text contained in the review the user gave

**Sentiment**: Sentiment of the review (Positive and Negative, Or 0 for Negative, 1 for positive)

**Train.csv**:  Labelled tweets on which to train your model
         
The Models I fine-tuned include: \n
- Roberta: Achieving an Accuracy score of 0.94 but did overfit \n 
- Tiny Bert: Achieving an Accuracy scrore of 0.87 barely overfitted
             
         """)



data= pd.read_csv("datasets/Train.csv")

st.subheader("A sample of the orginal Dataframe (Train.csv)")

st.write(data.head())

st.subheader("A sample of the preprocessed dataset")

data_clean= pd.read_csv("datasets/capstone_data.csv")

data_clean= data_clean.drop("Unnamed: 0", axis= 1)

st.write(data_clean.head())