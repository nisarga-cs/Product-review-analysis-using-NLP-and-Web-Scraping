import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
#import spacy
from scipy.sparse import csr_matrix
from nltk.stem.snowball import stopwords
import joblib

nltk.download('stopwords')
stopwords.words('english')
set_stopwords = set(stopwords.words('english'))
set_stopwords.discard('no')
set_stopwords.discard('not')

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet = nltk.WordNetLemmatizer()
#spacy_lem = spacy.load("en_core_web_sm")

data = pd.read_csv("C:\\Users\\Admin\\Downloads\\AmazonReview.csv",encoding="utf-8")
#df
data.loc[data['Sentiment']<=3,'Sentiment'] = 0
data.loc[data['Sentiment']>3,'Sentiment'] = 1

import nltk
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

review_data = []
for text1 in data['Review']:
  #convert to lower case
  text = str(text1)
  text = text.lower()
  #Remove non alphabetic characters and non space characters
  text = re.sub('[^a-z ]','',text)
  #Convert string to list
  list_words = text.split(' ')
  #Remove Stopwords
  text = []
  for word in list_words:
    if word not in set_stopwords:
      text.append(porter.stem(word))
  #Convert list back to string
  text = ' '.join(text)
  review_data.append(text)

review_data

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
x = tfidfvec.fit_transform(review_data)

pd.DataFrame.sparse.from_spmatrix(x)

tfidfvec.vocabulary_

y = data['Sentiment'].values
y = y.reshape(-1,1)

print(x.shape)
print(y.shape)


tfi = TfidfVectorizer(max_features=26493)
csr_matrix((1, 5949), dtype = np.int8).toarray()
tfi.fit_transform(review_data)

joblib.dump(tfi, "C:\\Users\\Admin\\OneDrive\\Desktop\\Py Folder\\Review Analysis Using NLP\\fitted_tfidf_vectorizer264.pkl")#saving vectorizer
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,
                                                 random_state = 2022)

from sklearn.svm import SVC
rfc = SVC(C=0.6,gamma=0.5)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

model_save_path = "C:\\Users\\Admin\\OneDrive\\Desktop\\Py Folder\\Review Analysis Using NLP\\sentimentAnalysisModelRFC.pkl"  
joblib.dump(rfc, model_save_path)

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rfc.classes_)
disp.plot()
plt.show()
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision for label 0:",precision_score(y_test,y_pred,pos_label=0,average=None))
print("Recall for label 0:",recall_score(y_test,y_pred,pos_label=0,average=None))
print("F-Score:",f1_score(y_test,y_pred,average=None))

