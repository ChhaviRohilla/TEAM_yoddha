import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv('messages.csv')

# Split data
data_train, data_test, labels_train, labels_test = train_test_split(df.text, df.type, test_size=0.2, random_state=0)

# Vectorize text
vectorizer = CountVectorizer()
data_train_count = vectorizer.fit_transform(data_train)
data_test_count = vectorizer.transform(data_test)

# Train model
clf = MultinomialNB()
clf.fit(data_train_count, labels_train)

# Evaluate model
predictions = clf.predict(data_test_count)
print("Confusion Matrix:\n", confusion_matrix(labels_test, predictions))
print("Classification Report:\n", classification_report(labels_test, predictions))

# Save model and vectorizer
joblib.dump(clf, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')