from flask import Flask, render_template, request,redirect
import joblib
import re
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

Fake_news = Flask(__name__,template_folder='./templates',static_folder='./static')

vector = joblib.load(r"Models\TfidfVectorizer.pkl")
loaded_model = joblib.load(r"Models\PassiveAggressiveClassifier.pkl")
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
corpus = []

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
     
    return prediction



@Fake_news.route('/',methods=['GET'])
def home():
    return redirect("http://localhost:8502")
    
    
@Fake_news.route('/Model_prediction',methods=["POST"])
def predict():
    Data=request.get_json()
    text=Data.get('txt',{})
    if text:
        pred=fake_news_det(text)
        if pred[0] == 1:
            res="Prediction of the News :  Looking Fake News📰"
        else:
            res="Prediction of the News : Looking Real News📰 "
    else:
        return "Input is empty",404        
    return res
     
    
if __name__ =="__main__":
    Fake_news.run('0.0.0.0',5000)    
    
