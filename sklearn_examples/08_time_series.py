import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)

# Generate time series with trend, seasonality, and noise
trend = np.linspace(0, 100, n_samples)
seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
noise = np.random.normal(0, 5, n_samples)
values = trend + seasonality + noise

df = pd.DataFrame({
    'date': dates,
    'value': values
})

# Feature engineering for time series
def create_features(df):
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    
    # Lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)
    df['lag_30'] = df['value'].shift(30)
    
    # Rolling features
    df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
    df['rolling_std_7'] = df['value'].rolling(window=7).std()
    
    return df

# Create features
df_features = create_features(df)
df_features = df_features.dropna()

# Prepare data for modeling
X = df_features.drop(['date', 'value'], axis=1)
y = df_features['value']

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    mse_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
    
    results[name] = mse_scores
    print(f"{name} - Mean MSE: {np.mean(mse_scores):.2f}")

# Visualize results
plt.figure(figsize=(15, 5))

# Original time series
plt.subplot(131)
plt.plot(df['date'], df['value'])
plt.title('Original Time Series')
plt.xticks(rotation=45)

# Feature importance (for Random Forest)
plt.subplot(132)
rf_model = models['Random Forest']
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=importance, x='importance', y='feature')
plt.title('Feature Importance')

# Cross-validation results
plt.subplot(133)
plt.boxplot([results[name] for name in models.keys()], labels=models.keys())
plt.title('Model Comparison')
plt.ylabel('MSE')

plt.tight_layout()
plt.show() 