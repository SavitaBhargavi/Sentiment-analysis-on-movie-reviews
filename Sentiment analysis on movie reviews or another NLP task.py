# -------------------------------
# Complete Sentiment Analysis Project (NLTK Movie Reviews)
# -------------------------------

import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import random
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------------------
# Step 1: Download NLTK datasets
# -------------------------------
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# -------------------------------
# Step 2: Load and shuffle movie reviews
# -------------------------------
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]
random.shuffle(reviews)

# -------------------------------
# Step 3: Preprocess text
# -------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(words):
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

processed_reviews = [(preprocess_text(words), category) for words, category in reviews]

# -------------------------------
# Step 4: Prepare features and target
# -------------------------------
X = [review[0] for review in processed_reviews]
y = [review[1] for review in processed_reviews]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vectors = vectorizer.fit_transform(X)  # keep sparse for SVC

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 5: Train SVM classifier
# -------------------------------
classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train, y_train)

# -------------------------------
# Step 6: Predict and evaluate
# -------------------------------
y_pred = classifier.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# Step 7: Predict on entire dataset for visualization / Power BI
# -------------------------------
predicted_sentiments = classifier.predict(X_vectors)

# Save results
results_df = pd.DataFrame({
    "Review": X,
    "Actual_Sentiment": y,
    "Predicted_Sentiment": predicted_sentiments
})
results_df.to_csv("sentiment_analysis_results.csv", index=False)
print("Results saved to sentiment_analysis_results.csv")

# -------------------------------
# Step 8: Sentiment Distribution Visualization
# -------------------------------
sentiment_counts = pd.Series(predicted_sentiments).value_counts()

plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green','red'])
plt.title("Predicted Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# -------------------------------
# Step 9: Word Clouds for Positive & Negative Reviews
# -------------------------------
all_positive_text = ' '.join([X[i] for i in range(len(X)) if predicted_sentiments[i]=='pos'])
all_negative_text = ' '.join([X[i] for i in range(len(X)) if predicted_sentiments[i]=='neg'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(all_positive_text)
wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(all_negative_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews Word Cloud")
plt.savefig("positive_wordcloud.png")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews Word Cloud")
plt.savefig("negative_wordcloud.png")
plt.show()

