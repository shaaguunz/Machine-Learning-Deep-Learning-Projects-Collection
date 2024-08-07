from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import joblib
from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer()
nltk.download('wordnet')
from scipy.stats import mode
from sklearn.feature_extraction.text import TfidfVectorizer
tf = joblib.load('tf')
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)   #remove punctuation , numbers, links , synmbols
    text = re.sub('\n', '', text) #replacing new line with ''
    text = [word for word in text.split(' ') if word not in stopwords]  #removing stop words
    text=" ".join(text)  #joining text for lemmatization
    
    text=word_tokenize(text)
    return text

model = joblib.load('model')
model1 = joblib.load('model1')
model2 = joblib.load('model2')

stopwords=stopwords.words('english')
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message = request.form['text']

        vect = tf.transform([message])
        print(vect)
        
        #lets create majority voting using 3 models and here we are using 3 models 
        

        predictions = np.array([model.predict(vect), model1.predict(vect)])
        prediction = mode(predictions).mode[0]
        
      
        



    return render_template('result.html',pred = prediction,msg=message)






if __name__=='__main__':
    app.run(debug=True)
