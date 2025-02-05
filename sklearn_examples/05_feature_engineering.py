from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)
print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {X_poly.shape[1]}")

# 2. Feature Selection with SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
selected_features_mask = selector.get_support()

# Plot feature scores
plt.figure(figsize=(12, 5))
plt.bar(range(X_train.shape[1]), selector.scores_)
plt.xlabel('Feature Index')
plt.ylabel('F-score')
plt.title('Feature Importance Scores')
plt.show()

# 3. Recursive Feature Elimination (RFE)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)

# Plot RFE selected features
plt.figure(figsize=(12, 5))
plt.bar(range(X_train.shape[1]), rfe.ranking_)
plt.xlabel('Feature Index')
plt.ylabel('Ranking')
plt.title('RFE Feature Rankings')
plt.show()

# 4. Feature Importance from Random Forest
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 5))
plt.bar(range(X_train.shape[1]), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# Compare performance with different feature selection methods
results = {}
for name, X_features in [
    ('Original', X_train),
    ('Polynomial', X_poly),
    ('SelectKBest', X_selected),
    ('RFE', X_rfe)
]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_features, y_train)
    score = rf.score(X_test, y_test)
    results[name] = score
    print(f"{name} features accuracy: {score:.4f}") 