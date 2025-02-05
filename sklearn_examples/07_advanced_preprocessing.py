from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset with various types of features
np.random.seed(42)
n_samples = 1000

# Create DataFrame with mixed data types and missing values
data = {
    'numeric_1': np.random.normal(0, 1, n_samples),
    'numeric_2': np.random.normal(10, 5, n_samples),
    'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
    'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
    'ordinal': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'binary': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)

# Add missing values
for col in df.columns:
    mask = np.random.random(n_samples) < 0.1
    df.loc[mask, col] = np.nan

# Define feature types
numeric_features = ['numeric_1', 'numeric_2']
categorical_features = ['category_1', 'category_2']
ordinal_features = ['ordinal']
binary_features = ['binary']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Medium')),
    ('encoder', LabelEncoder())
])

# Combine all transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split data
X = df.drop('binary', axis=1)
y = df['binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit and evaluate pipeline
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Visualize preprocessing effects
plt.figure(figsize=(15, 5))

# Original numeric distribution
plt.subplot(131)
sns.histplot(df['numeric_1'].dropna(), bins=30)
plt.title('Original Distribution')

# Scaled numeric distribution
scaled_data = pipeline.named_steps['preprocessor'].transform(X_train)
plt.subplot(132)
sns.histplot(scaled_data[:, 0], bins=30)
plt.title('Scaled Distribution')

# Missing values handling
plt.subplot(133)
missing_counts = df.isnull().sum()
sns.barplot(x=missing_counts.index, y=missing_counts.values)
plt.xticks(rotation=45)
plt.title('Missing Values by Feature')

plt.tight_layout()
plt.show() 