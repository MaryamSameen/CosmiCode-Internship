import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Example data (replace with your own dataset for real use)
texts = [
    "I love this movie", "This film was terrible", "Amazing experience", "Worst movie ever",
    "I enjoyed the film", "Not my favorite", "Absolutely fantastic", "I hated it"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Tokenize text
max_words = 1000
max_len = 10
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Download GloVe embeddings (use 50d for demo)
# Download from: https://nlp.stanford.edu/data/glove.6B.zip and extract 'glove.6B.50d.txt'
embedding_index = {}
with open("glove.6B.50d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_dim = 50
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build model with embedding and RNN
model = keras.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix],
                     input_length=max_len, trainable=False),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test), verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)