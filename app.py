import streamlit as st
import joblib

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import pickle



# Open the file in binary read mode
with open("Emailclassifier_model.pkl", "rb") as file:
    classifier_model = pickle.load(file)

wordvect_model = Word2Vec.load("word2vec_model.model")

def preprocess_sentence(sentence):
    return simple_preprocess(sentence)

def vectorize_sentence(tokens, wordvect_model):
    tokens = [token for token in tokens if token in wordvect_model.wv]
    print('Tokens are ----------------',tokens)
    if not tokens:
        return np.zeros(wordvect_model.vector_size)
    print('avaergae', np.mean([wordvect_model.wv[token] for token in tokens], axis=0))
    return np.mean([wordvect_model.wv[token] for token in tokens], axis=0)

def predict_sentence(sentence, wordvect_model, classifier):
    tokens = preprocess_sentence(sentence)
    vector = vectorize_sentence(tokens, wordvect_model)
    prediction = classifier.predict([vector])
    return "spam" if prediction == 1 else "ham"

st.title("Email Spam Classifier")
st.write("Enter the email content below:")

user_input = st.text_area("Email Content")

if st.button("Classify"):
    result = predict_sentence(user_input, wordvect_model, classifier_model)
    if result == "spam":
        st.markdown(f"<p style='color:red; font-size:20px; font-weight:bold;'>The email is classified as: {result}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:green; font-size:20px; font-weight:bold;'>The email is classified as: {result}</p>", unsafe_allow_html=True)

