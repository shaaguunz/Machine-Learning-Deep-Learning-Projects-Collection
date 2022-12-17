from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
import joblib

def preprocess(t):
    text = str(text).lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)   #remove punctuation , numbers, links , synmbols
    text = re.sub('\n', '', text) #replacing new line with ''
    text = [word for word in text.split(' ') if word not in stopword]  #removing stop words
    text=" ".join(text)  #joining text for lemmatization
    text = [lemmatize.lemmatize(word) for word in text.split(' ')]  #lemmatizing words
    text=" ".join(text) #again joining text for easy tokenization
    
    text=word_tokenize(text)
    return text

cv = joblib.load('cv')
model = joblib.load('model')


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message = request.form['text']
        message = preprocess(message)

        vect = cv.transform(message).toarray()
        prediction = model.predict(vect)


    return render_template('result.html',pred = prediction,msg=message)






if __name__=='__main__':
    app.run(debug=True)
