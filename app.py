# Summurizing the processes which could be used for web application
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib as jb
import warnings

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🕊")
st.title("Twitter Sentimental Analysis")
warnings.filterwarnings("ignore")

container="""
<style>
[data-baseweb="base-input"]{
margin-top: 40px; 
margin-bottom: 40px; 
}
</style>
"""
st.markdown(container,unsafe_allow_html=True)

red="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#FF0000;
color: #ffffff;
}
[data-testid="stMarkdownContainer"]{
color: #ffffff;
}
[class="css-zt5igj e16nr0p33"]{
color: #ffffff;
}
</style>
"""
blue="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#2B65EC;
color: #ffffff;
}
[data-testid="stMarkdownContainer"]{
color: #ffffff;
}
[class="css-zt5igj e16nr0p33"]{
color: #ffffff;
}
</style>
"""
green="""
<style>
[data-testid="stAppViewContainer"]{
background-color:#008000;
color: #ffffff;
}
[data-testid="stMarkdownContainer"]{
color: #ffffff;
}
[class="css-zt5igj e16nr0p33"]{
color: #ffffff;
}
</style>
"""

tsa = jb.load('tsa.joblib')
model = tsa['model']
bow_vectorizer = tsa['bow_vectorizer']
input_tweet= st.text_input("Enter the tweet here")
df = pd.DataFrame([{'tweet':input_tweet}])
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]"," ")
df['clean_tweet'] = df['clean_tweet'].apply(lambda x:" ".join([w for w in x.split() if len(w)>3]))
tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=" ".join(tokenized_tweet[i])
df['clean_tweet']=tokenized_tweet
bow = bow_vectorizer.transform(df['clean_tweet'])
pred = model.predict(bow)
if input_tweet=="":
    pass
elif pred==1:
    st.write("### This is a Positive Tweet")
    st.markdown(green, unsafe_allow_html=True)
elif pred==0:
    st.write("### This is a Neutral Tweet")
    st.markdown(blue, unsafe_allow_html=True)
elif pred==-1:
    st.write("### This is a Negative Tweet")
    st.markdown(red, unsafe_allow_html=True)
st.write("\n\n\n\n\n\n")
with st.container():
    left, middle, right = st.columns(3)
    with left:
        st.markdown("<a href='https://www.linkedin.com/in/prasad-posture-6a3a77215/' target='blank'><img align='center' src='https://img.shields.io/badge/-Prasad Posture-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/prasad-posture-6a3a77215/' alt='Prasad Posture' height='20' width='100' /></a>", unsafe_allow_html=True)
    with middle:
        st.markdown("<a href='https://github.com/prasadposture' target='blank'><img align='center' src='https://img.shields.io/badge/-prasadposture-black?style=flat-square&logo=GitHub&logoColor=white&link=https://github.com/prasadposture' alt='prasadposture' height='20' width='100' /></a>", unsafe_allow_html=True)
    with right:
        st.markdown("<a href='https://www.kaggle.com/prasadposture121' target='blank'><img align='center' src='https://img.shields.io/badge/-prasadposture121-blue?style=flat-square&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/prasadposture121' alt='prasadposture121' height='20' width='100' /></a>", unsafe_allow_html=True)
