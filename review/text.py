import streamlit as st
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import pickle
st.subheader("Predict the Amazon Review Score")
# st.text_input("Input your review:")
review = st.text_input("Review Summary:",placeholder="enter review summary")
review_desc = st.text_area("Review Description:",placeholder="enter your review")
stop_words = stopwords.words("english") + ["br", "html", "www", "k", "http"]

def set_unique_words(lem_text):
    res = lambda lem_text: ' '.join(set(row_word for row_word in lem_text.split(" ") if row_word not in stop_words))
    return res(lem_text)
def striphtml(lem_text):
    p = re.compile(r'<.*?>hhtp//+s|www\.\S+')
    return p.sub('', lem_text)
import string
punct_to_remove=string.punctuation
def remove_punctuation(lem_text):
    return lem_text.translate(str.maketrans('','',punct_to_remove))
    
def load_module(lem_text):
    if int(len(lem_text)) != 1:
        # st.write(len(lem_text))
        rf_bow_amazon_rev = pickle.load(open(r"C:\Users\Dell\Desktop\ai6\nlp\vocab.pkl", "rb"))
        rf_amazon_rev = pickle.load(open(r"C:\Users\Dell\Desktop\ai6\nlp\nb.pkl", "rb"))
        X_test_ = rf_bow_amazon_rev.transform([lem_text])  
        score = rf_amazon_rev.predict(X_test_.toarray().reshape(1, -1))
        st.write("Predicted Review Score:  {0}".format(score[0]))
def preprocess(final_text):
    final_text = list(map(lambda x:x.replace(":)", "good "),  final_text.split(" ")))
    final_text = list(map(lambda x:x.replace(":(", "bad "),  final_text))
    final_text_re = re.sub("[^a-zA-Z]", " ", str(final_text).lower())
    lemmatize = WordNetLemmatizer() # stemmer = PorterStemmer()
    lem_text = [lemmatize.lemmatize(x.lower()) for x in final_text_re.split(" ") if x not in stop_words]
    lem_text = [lemmatize.lemmatize(x) for x in lem_text if x!=""]
    lem_text = " ".join([x for x in lem_text])
    return lem_text

if st.button("Predict Review !!", help="Predict the \"Amazon Product Review\" "):
    final_text = review + " " + review_desc
    #final_text=review_desc  
    # st.write("No of chars: {0}".format( int(len(final_text))-1))
    if int(len(final_text)) == 1:
        st.success("Please input the Review")
    else:
        lem_text =  preprocess(final_text)
        lem_text = set_unique_words(lem_text)
        # st.write(lem_text)
        load_module(lem_text)
