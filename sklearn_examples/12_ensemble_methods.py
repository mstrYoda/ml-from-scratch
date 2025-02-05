from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Generate dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    n_redundant=5, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_score = rf.score(X_test_scaled, y_test)

# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_score = gb.score(X_test_scaled, y_test)

# 3. AdaBoost
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    random_state=42
)
ada.fit(X_train_scaled, y_train)
ada_score = ada.score(X_test_scaled, y_test)

# 4. Voting Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train_scaled, y_train)
voting_score = voting_clf.score(X_test_scaled, y_test)

# 5. Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_clf.fit(X_train_scaled, y_train)
stacking_score = stacking_clf.score(X_test_scaled, y_test)

# Print results
print("Model Scores:")
print(f"Random Forest: {rf_score:.4f}")
print(f"Gradient Boosting: {gb_score:.4f}")
print(f"AdaBoost: {ada_score:.4f}")
print(f"Voting Classifier: {voting_score:.4f}")
print(f"Stacking Classifier: {stacking_score:.4f}")

# Visualizations
plt.figure(figsize=(15, 5))

# Feature importance comparison
plt.subplot(131)
importances = pd.DataFrame({
    'Random Forest': rf.feature_importances_,
    'Gradient Boosting': gb.feature_importances_,
    'AdaBoost': ada.feature_importances_
})
sns.boxplot(data=importances)
plt.title('Feature Importance Distribution')
plt.xticks(rotation=45)

# Model comparison
plt.subplot(132)
model_scores = {
    'RF': rf_score,
    'GB': gb_score,
    'ADA': ada_score,
    'Voting': voting_score,
    'Stacking': stacking_score
}
plt.bar(model_scores.keys(), model_scores.values())
plt.title('Model Comparison')
plt.ylim(0.8, 1.0)

# Learning curves for Gradient Boosting
plt.subplot(133)
plt.plot(gb.train_score_, label='Training')
plt.plot(gb.validation_score_, label='Validation')
plt.title('Gradient Boosting Learning Curves')
plt.legend()

plt.tight_layout()
plt.show() 