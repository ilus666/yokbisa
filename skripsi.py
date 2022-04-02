import joblib
import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


st.title('Prediksi Sentiment')
st.subheader('Implementasi Sentiment Analysis Berdasarkan Tweets Masyarakat Terhadap Kinerja Presiden dalam Aspek Penanganan Covid-19')
st.text('Algoritma SVM OneVSRest')

#input
my_form = st.form(key="form1")
name = my_form.text_input(label = "Masukkan teks berbahasa indonesia:")
submit = my_form.form_submit_button(label = 'submit')
teks = name.title()
model = joblib.load(open('model_B.pkl', 'rb'))
tfidf = joblib.load(open('tf_idf_B.pkl', 'rb'))

def preprocessing(kata):
    kata = data_cleaning(kata)
    kata = case_folding(kata)
    kata = tokenizing(kata)
    kata = rejoin(kata)
    return kata

def data_cleaning(kata):
    data_clean = kata
    data_clean = re.sub(r'\d+', " ", data_clean)  #remove angka
    data_clean = re.sub(r'http\S+', " ", data_clean)  #hapus url
    data_clean = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", data_clean)# Remove simbol, angka dan karakter aneh
    return data_clean

def case_folding(kata):
    case_fold=kata.lower()
    return case_fold

def tokenizing(kata):
    tokenisasi=word_tokenize(kata)
    return tokenisasi

def rejoin(kata):
    rejoin_teks=" ".join(kata)
    rejoin_teks=re.sub(r'\d+', '',  rejoin_teks) #remove number **
    return rejoin_teks

kalimat = preprocessing(teks)
data = tfidf.transform([kalimat])
hasil = model.predict(data)
hasil1 = " ".join(hasil)
hasil_proba = model.predict_proba(data)

if submit:
     st.write('Hasil kelas emosinya adalah :',hasil1)