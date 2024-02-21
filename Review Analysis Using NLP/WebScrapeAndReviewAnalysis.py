import requests, json
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

url = 'https://mamaearth.in/product/glow-serum-foundation-almond-glow'
r = requests.get(url)
soup=BeautifulSoup(r.content,'lxml')


stp_words=stopwords.words('english')
def clean_review(review):
   review = review.lower()  # Convert to lowercase
   review = re.sub(r'[^\w\s]', '', review)  # Remove punctuation
   review = " ".join(word for word in review.split() if word not in stopwords.words('english'))  # Remove stop words  
   return review


model_save_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\Py Folder\\Review Analysis Using NLP\\sentimentAnalysisModelRFC.pkl"  
loaded_model = joblib.load(model_save_path)   
    

data = json.loads(soup.find('script', type='application/ld+json').text)
reviews = data['review']
print(reviews)
print(type(reviews))
for r in reviews:
    curr_review=r.get('reviewBody')
    clean_review(curr_review)
    print(curr_review)  
    vectorizer = joblib.load("C:\\Users\\Admin\\OneDrive\\Desktop\\Py Folder\\Review Analysis Using NLP\\fitted_tfidf_vectorizer264.pkl")
    vector = vectorizer.transform([curr_review])
    prediction = loaded_model.predict(vector)
    print("pred is ",prediction)
    print(f"Predicted sentiment: {prediction[0]} (0: negative, 1: positive)")
    print("-----------------")

    
