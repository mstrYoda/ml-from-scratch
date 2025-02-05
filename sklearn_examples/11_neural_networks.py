from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Classification with Neural Network
# Generate classification dataset
X_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=15, 
    n_redundant=5, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clf)
X_test_scaled = scaler.transform(X_test_clf)

# Create and train classifier
clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    verbose=True
)

clf.fit(X_train_scaled, y_train_clf)

# Evaluate classifier
y_pred_clf = clf.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

# 2. Regression with Neural Network
# Generate regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=0.1, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale the data
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Create and train regressor
reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    verbose=True
)

reg.fit(X_train_reg_scaled, y_train_reg)

# Evaluate regressor
y_pred_reg = reg.predict(X_test_reg_scaled)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"\nRegression MSE: {mse:.4f}")

# Visualizations
plt.figure(figsize=(15, 5))

# Plot learning curves for classifier
plt.subplot(131)
plt.plot(clf.loss_curve_)
plt.title('Classification Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Plot learning curves for regressor
plt.subplot(132)
plt.plot(reg.loss_curve_)
plt.title('Regression Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Plot regression predictions vs actual
plt.subplot(133)
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 
         'r--', lw=2)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show() 