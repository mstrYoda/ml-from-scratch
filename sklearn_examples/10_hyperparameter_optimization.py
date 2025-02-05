from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 1. Grid Search CV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Score:", grid_search.best_score_)

# 2. Random Search CV
param_dist = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__max_depth': randint(10, 50),
    'classifier__min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1
)
random_search.fit(X_train, y_train)

print("\nRandom Search Best Parameters:", random_search.best_params_)
print("Random Search Best Score:", random_search.best_score_)

# 3. Optuna Optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
    }
    
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(**params))
    ])
    
    return cross_val_score(clf, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("\nOptuna Best Parameters:", study.best_params)
print("Optuna Best Score:", study.best_value)

# Visualization of results
plt.figure(figsize=(15, 5))

# Grid Search Results
plt.subplot(131)
results = pd.DataFrame(grid_search.cv_results_)
sns.scatterplot(data=results, x='mean_test_score', y='mean_fit_time')
plt.title('Grid Search Results')

# Random Search Results
plt.subplot(132)
results = pd.DataFrame(random_search.cv_results_)
sns.scatterplot(data=results, x='mean_test_score', y='mean_fit_time')
plt.title('Random Search Results')

# Optuna Results
plt.subplot(133)
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optuna Optimization History')

plt.tight_layout()
plt.show() 