import pickle
load_model=pickle.load(open('imdb_sentiment.sav','rb'))
tfidf=pickle.load(open('tfidf_imdb_sentiment.pkl','rb'))

import string
exclude=string.punctuation
def punc(text):
    return text.translate(str.maketrans('','',exclude))

import nltk
from nltk.corpus import stopwords
s=stopwords.words('english')
def remove_stopwords(text):
    new_text=[]
    for w in text.split():
        if w in s:
            new_text.append('')
        else:
            new_text.append(w)
    return ' '.join(new_text)


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem_words(text):
    return ' '.join([ps.stem(word) for word in text.split()])

import streamlit as st
text=st.text_area("Write Your Review")

text=punc(str(text))
text=remove_stopwords(text)
text=stem_words(text)
text=tfidf.transform([text]).toarray()
if st.button("Check"):
    if load_model.predict(text) == 1:
        st.success('Positive Review')
    else:
        st.error('Negative Review')        
