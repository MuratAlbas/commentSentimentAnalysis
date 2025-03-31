import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('path/to/your/YoutubeCommentsDataSet.csv')

# Creating the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    # 1D Convolutional Layer with 64 filters, size of the kernel:5 and ReLu activation
    Conv1D(64, 5, activation='relu'),
    # applying max pooling
    GlobalMaxPooling1D(),
    # 3 output units for 3 sentiment and use of softmax for activation
    Dense(3, activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Ensuring all values in 'Comment' are strings
df['Comment'] = df['Comment'].fillna('').astype(str)

# Mapping sentiment labels to integers
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

# Creating tokenizer for maximum of 10000 words
tokenizer = Tokenizer(num_words=10000)
#Building the mapping of words to integer indices
tokenizer.fit_on_texts(df['Comment'])

# Converting text data into numerical data
sequences = tokenizer.texts_to_sequences(df['Comment'])
# Ensuring all sequences have the same length
padded_sequences = pad_sequences(sequences, maxlen=100)

# Training the model
model.fit(padded_sequences, df['Sentiment'], epochs=10, batch_size=32, validation_split=0.2)

#Evaluating the model
test_loss, test_acc = model.evaluate(padded_sequences, df['Sentiment'])
print(f"Accuracy: {test_acc:}")
