# Summurizing the processes which could be used for web application
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib as jb
import warnings
warnings.filterwarnings("ignore")
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
if pred==1:
    st.write("Positive Tweet")
elif pred==0:
    st.write("Neutral Tweet")
elif pred==-1:
    st.write("Negative Tweet")