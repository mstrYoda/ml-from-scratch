from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42,
    weights=[0.9, 0.1]  # Make it imbalanced
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Function to evaluate and plot results
def evaluate_model(y_true, y_pred, title):
    print(f"\n{title} Results:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

# 1. Baseline model (without handling imbalance)
print("Original dataset distribution:", Counter(y_train))
clf_base = RandomForestClassifier(random_state=42)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)
evaluate_model(y_test, y_pred_base, "Baseline")

# 2. SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nSMOTE resampled distribution:", Counter(y_train_smote))

clf_smote = RandomForestClassifier(random_state=42)
clf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = clf_smote.predict(X_test)
evaluate_model(y_test, y_pred_smote, "SMOTE")

# 3. ADASYN oversampling
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
print("\nADASYN resampled distribution:", Counter(y_train_adasyn))

clf_adasyn = RandomForestClassifier(random_state=42)
clf_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_adasyn = clf_adasyn.predict(X_test)
evaluate_model(y_test, y_pred_adasyn, "ADASYN")

# 4. Combined approach (SMOTE + Tomek links)
smote_tomek = SMOTETomek(random_state=42)
X_train_combined, y_train_combined = smote_tomek.fit_resample(X_train, y_train)
print("\nSMOTE + Tomek links distribution:", Counter(y_train_combined))

clf_combined = RandomForestClassifier(random_state=42)
clf_combined.fit(X_train_combined, y_train_combined)
y_pred_combined = clf_combined.predict(X_test)
evaluate_model(y_test, y_pred_combined, "SMOTE + Tomek")

# Compare ROC curves
plt.figure(figsize=(10, 6))
for name, clf, X_tr, y_tr in [
    ('Baseline', clf_base, X_train, y_train),
    ('SMOTE', clf_smote, X_train_smote, y_train_smote),
    ('ADASYN', clf_adasyn, X_train_adasyn, y_train_adasyn),
    ('SMOTE + Tomek', clf_combined, X_train_combined, y_train_combined)
]:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show() 