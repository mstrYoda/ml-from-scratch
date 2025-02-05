from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score,
    KFold, StratifiedKFold, ROC_curve, precision_recall_curve
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

# 1. Learning Curves
def plot_learning_curves(X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, scoring='accuracy'
    )
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. Validation Curves
def plot_validation_curves(X, y):
    param_range = np.logspace(-3, 3, 7)
    train_scores, val_scores = validation_curve(
        RandomForestClassifier(random_state=42),
        X, y, param_name='max_depth',
        param_range=param_range, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plt.figure(figsize=(10, 5))
    plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Training')
    plt.semilogx(param_range, np.mean(val_scores, axis=1), label='Validation')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Validation Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 3. ROC and PR Curves
def plot_roc_pr_curves(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(15, 5))
    
    # Plot ROC curve
    plt.subplot(131)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot PR curve
    plt.subplot(132)
    plt.plot(recall, precision, 
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(133)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

# Run all evaluations
plot_learning_curves(X, y)
plot_validation_curves(X, y)
plot_roc_pr_curves(X, y)

# Print classification report
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print("\nCross-validation scores:", scores)
print(f"Average CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})") 