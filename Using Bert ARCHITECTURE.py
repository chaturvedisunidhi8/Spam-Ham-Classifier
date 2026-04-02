import tensorflow as tf
import tensorflow_hub as hub  
import tensorflow_text as text 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("sam.csv")
print(df.head())

# Check null values and shape
print(df.isnull().sum())
print(df.shape)

# Separating input and output feature
X = df['message']
y = df['category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load BERT preprocess and encoder
bert_preprocess = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
bert_encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2")

# Function to get sentence embeddings
def get_sentence_embedding(sentence):
    preprocessed_text = bert_preprocess([sentence])
    return bert_encoder(preprocessed_text)['pooled_output']

#  embeddings & cosine similarity
emb1 = get_sentence_embedding("Discount, hurry up,get upto 70% off on groceries")
emb2 = get_sentence_embedding("wohoo!, you won the ticket for the europe ,hurry up offer is valid for while")
print("Cosine Similarity:", cosine_similarity(emb1, emb2))

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)
l = tf.keras.layers.Dropout(0.1)(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(l)

model = tf.keras.Model(inputs=[text_input], outputs=[l])


model.summary()

# Compile model
model.compile(optimizer='adam',         loss='binary_crossentropy',          metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10)

# Evaluate model
model.evaluate(X_test, y_test)

# Predict for a review
review = ["your loan has been approved ,provide details."]
print(model.predict(review))
