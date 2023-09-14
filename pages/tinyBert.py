
import streamlit as st 
import torch 
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
import numpy as np
import pandas as pd
import re
from scipy.special import softmax
from transformers import pipeline
import xformers
import requests
import json

from streamlit_lottie import st_lottie 
from wordcloud import WordCloud, STOPWORDS

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from transformers import pipeline

from num2words import num2words

##declaring my Stopwords
stopwords= STOPWORDS
## Front end
st.title("Welcome to the Fine-Tuned Tiny-Bert Sentiment Classifier  Page")

##including an animation to my page 

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
# initializaing my session state 
if 'lottie_hello_2' not in st.session_state:
    st.session_state.lottie_hello_2 = load_lottiefile("./lottie_animations/dsbert.json")
# creating a funciton to upload the file while implementing session state
def handle_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        st.session_state.lottie_hello_2 = load_lottiefile(uploaded_file.name)
# displaying the Lottie animation
st_lottie(st.session_state.lottie_hello_2, height=200)

text = st.text_input("Please Enter Your Thoughts of The Movie: ")
## Creating a cache to store my model for efficiency
@st.cache_data(ttl=86400)
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model
## Creating my tokenizer
@st.cache_data(ttl=86400)
def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer
## Cleaning
def clean_text(text):
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords])
    # Convert numbers to words
    text = " ".join([num2words(word) if word.isdigit() else word for word in text.split()])
    # Remove special characters
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    ##for numbers that do no get converted will be dropped 
    text = re.sub(r'\d+', '', text)
    return text
## Running my input through my function
text_input = clean_text(text)
if 'ro_model' not in st.session_state:
    st.session_state.ro_model = load_model("gArthur98/Capstone_TinyBert")
if 'ro_token' not in st.session_state:
    st.session_state.ro_token = load_tokenizer("gArthur98/Capstone_TinyBert")
pipe = pipeline("sentiment-analysis", model=st.session_state.ro_model, tokenizer=st.session_state.ro_token)

result = pipe(text_input)

final = st.button("Predict Sentiment")

## Initializing my session state
if final:
    for results in result:
        if results['label'] == 'LABEL_0':
            st.write(f"Your sentiment is Negative with a confidence score of {results['score']}")
        else:
           st.write(f"Your sentiment is Positive with a confidence score of {results['score']}")



st.write("""Example of sentences to input:
         
         - I hate this movie so bad \n
    - I love the movies scenes \n
    - The movie was very boring
         """)