from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.2, random_state=42
)

# 1. Basic Count Vectorization
count_vec = CountVectorizer(max_features=1000, stop_words='english')
X_count = count_vec.fit_transform(X_train)

# 2. TF-IDF Vectorization
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = tfidf_vec.fit_transform(X_train)

# Create pipelines
pipelines = {
    'Count Vectorizer': Pipeline([
        ('vectorizer', CountVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ]),
    'TF-IDF': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
}

# Train and evaluate models
results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

# Visualize top terms for each category
def plot_top_terms(vectorizer, feature_names, n_top_terms=10):
    plt.figure(figsize=(15, 10))
    for i, category in enumerate(categories):
        top_indices = np.argsort(vectorizer.idf_)[:n_top_terms]
        top_terms = [feature_names[i] for i in top_indices]
        plt.subplot(2, 2, i+1)
        plt.barh(range(n_top_terms), sorted(vectorizer.idf_)[:n_top_terms])
        plt.yticks(range(n_top_terms), top_terms)
        plt.title(f'Top terms in {category}')
    plt.tight_layout()
    plt.show()

plot_top_terms(tfidf_vec, tfidf_vec.get_feature_names()) 