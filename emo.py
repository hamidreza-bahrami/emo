import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
import time
from googletrans import Translator
import nltk
nltk.download('stopwords')

vector = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))
translator = Translator()

def stemming(content):
  con = re.sub('[^a-zA-Z]', ' ', content)
  con = con.lower()
  con = con.split()
  con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
  con = ' '.join(con)
  return con

def thought(text):
  text = stemming(text)
  input_text = [text]
  vector1 = vector.transform(input_text)
  prediction = load_model.predict(vector1)
  return prediction

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>مدل تشخیص احساس از متن</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>تشخیص بر اساس تحلیل متن کاربر</h2>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h4>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")

    text = st.text_area('متن خود را وارد کنید',height=None,max_chars=None,key=None)
    
    if st.button('تحلیل احساس متن'):
        if text == "":
            with st.chat_message("assistant"):
                with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                    time.sleep(3)
                    st.success(u'\u2713''تحلیل انجام شد')
                    st.write("<h4 style='text-align: right; color: gray;'>لطفا متن خود را بنویسید تا بتوانم تحلیل کنم</h4>", unsafe_allow_html=True)
        
        else:
            out = translator.translate(text)
            prediction_class = thought(out.text)
            if prediction_class == 'neutral':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس خنثی است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما از نظر احساسی دارای وزن یکسان هستند</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>در نتیجه حس خاصی از این متن درک نمی کنم</h4>", unsafe_allow_html=True)
            
            elif prediction_class == 'joy':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس مثبت است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس شادابی، عشق و انرژی هستند</h4>", unsafe_allow_html=True)

            elif prediction_class == 'sadness':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس منفی است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس غم،فقدان و ناامیدی هستند</h4>", unsafe_allow_html=True)

            elif prediction_class == 'fear':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس ترس است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس ترس و نگرانی هستند</h4>", unsafe_allow_html=True)

            elif prediction_class == 'surprise':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس هیجان و شگفت زدگی است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس غافلگیر شدن و بهت زده شدن هستند</h4>", unsafe_allow_html=True)

            elif prediction_class == 'anger':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس عصبانیت است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس خشم، کینه و نفرت هستند</h4>", unsafe_allow_html=True)

            elif prediction_class == 'shame':
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس شرمندگی است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس شرمندگی و شرمساری هستند</h4>", unsafe_allow_html=True)
    
            else:
                with st.chat_message("assistant"):
                    with st.spinner('''درحال بررسی، لطفا صبور باشید'''):
                        time.sleep(3)
                        st.success(u'\u2713''تحلیل انجام شد')
                        st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من، متن شما دارای حس انزجار است</h4>", unsafe_allow_html=True)
                        st.write("<h4 style='text-align: right; color: gray;'>کلمات متن شما ترکیبی از احساساتی نظیر حس انزجار، بیزاری و بی میلی هستند</h4>", unsafe_allow_html=True)
    else:
        pass
            
show_page()
