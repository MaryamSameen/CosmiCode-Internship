import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels
df['label_num'] = df.label.map({'ham':0, 'spam':1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42)

# Build and train pipeline with TfidfVectorizer
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'spam_classifier.pkl')

# Print accuracy
print(f"Test Accuracy with TfidfVectorizer: {pipeline.score(X_test, y_test):.2f}")
