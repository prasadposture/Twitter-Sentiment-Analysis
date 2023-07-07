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