#  Install and import libraries
!pip install gensim    
!pip install tqdm    
    
import gensim  
from gensim.models import Word2Vec, KeyedVectors
from gensim.downloader import load as api      

import pandas as pd
import numpy as np
import re
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tqdm import tqdm

#  Load pretrained Word2Vec model
wv = api('word2vec-google-news-300')  # Pretrained Google model (300 dimensions)

#  Read the dataset
message = pd.read_csv('spam.csv')

#  Text preprocessing and lemmatization 
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    corpus.append(review)


words = []
for sent in corpus:
    sent_token = sent_tokenize(' '.join(sent))
    for s in sent_token:
        words.append(s)

model = gensim.models.Word2Vec(words)
model.wv.index_to_key
model.corpus_count
model.epochs
model.wv.similar_by_word("good")
model.wv['good'].shape

# Average word vectors for each sentence
def avg_word2vec(doc):
    return np.mean([wv[word] for word in doc if word in wv.index_to_key], axis=0)


X = []
for i in tqdm(range(len(corpus))):
    X.append(avg_word2vec(corpus[i]))

# Independent and dependent variables
X_new = np.array(X)
print(X_new.shape)
print(X_new[0])
print(X_new[0].shape)

y = message['label'].apply(lambda x: 1 if x == 'spam' else 0)
y = pd.get_dummies(y, drop_first=True).values
print(message.shape)


df = pd.DataFrame()
for i in range(0, len(X)):
    df = df.append(pd.DataFrame(X[i].reshape(1, -1)), ignore_index=True)

print(df.shape) 

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

#  Train model
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make prediction
y_pred = classifier.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
