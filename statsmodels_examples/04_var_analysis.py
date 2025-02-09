import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
import seaborn as sns

# Generate multivariate time series data for system monitoring
np.random.seed(42)

def generate_system_data(n_samples=1000):
    # Generate time series with interdependencies
    
    # Temperature affects pressure and vibration
    temperature = np.random.normal(60, 5, n_samples)
    temperature = pd.Series(temperature).rolling(window=5).mean().fillna(method='bfill')
    
    # Pressure depends on temperature
    pressure = 2 * temperature + np.random.normal(100, 10, n_samples)
    pressure = pd.Series(pressure).rolling(window=3).mean().fillna(method='bfill')
    
    # Vibration depends on both temperature and pressure
    vibration = 0.3 * temperature + 0.2 * pressure + np.random.normal(0, 2, n_samples)
    vibration = pd.Series(vibration).rolling(window=2).mean().fillna(method='bfill')
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    data = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration
    }, index=dates)
    
    return data

# Generate data
system_data = generate_system_data()

# 1. Data Visualization
plt.figure(figsize=(15, 10))
for i, column in enumerate(system_data.columns, 1):
    plt.subplot(3, 1, i)
    plt.plot(system_data.index, system_data[column])
    plt.title(f'{column} Over Time')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(system_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. Stationarity Test
def check_stationarity(data):
    results = {}
    for column in data.columns:
        adf_test = adfuller(data[column])
        results[column] = {
            'ADF Statistic': adf_test[0],
            'p-value': adf_test[1],
            'Critical values': adf_test[4]
        }
    return results

stationarity_results = check_stationarity(system_data)
print("\nStationarity Test Results:")
for var, result in stationarity_results.items():
    print(f"\n{var}:")
    print(f"ADF Statistic: {result['ADF Statistic']:.4f}")
    print(f"p-value: {result['p-value']:.4f}")

# 4. VAR Model Fitting
model = VAR(system_data)
results = model.fit(maxlags=15, ic='aic')

print("\nVAR Model Summary:")
print(results.summary())

# 5. Granger Causality
print("\nGranger Causality Tests:")
for column in system_data.columns:
    granger_test = results.test_causality(column, system_data.columns.drop(column))
    print(f"\nGranger causality of {column}:")
    print(f"Test statistic: {granger_test.test_statistic:.4f}")
    print(f"p-value: {granger_test.pvalue:.4f}")

# 6. Impulse Response Analysis
irf = results.irf(periods=20)
irf.plot(orth=True)
plt.suptitle('Impulse Response Functions')
plt.tight_layout()
plt.show()

# 7. Forecast
nobs = 50
forecast = results.forecast(system_data.values[-nobs:], steps=24)
forecast_index = pd.date_range(
    start=system_data.index[-1],
    periods=25,
    freq='H'
)[1:]

forecast_df = pd.DataFrame(
    forecast,
    index=forecast_index,
    columns=system_data.columns
)

# Plot forecasts
plt.figure(figsize=(15, 10))
for i, column in enumerate(system_data.columns, 1):
    plt.subplot(3, 1, i)
    plt.plot(system_data.index[-100:], system_data[column][-100:], label='Observed')
    plt.plot(forecast_df.index, forecast_df[column], 'r--', label='Forecast')
    plt.title(f'{column} Forecast')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Model Diagnostics
# Durbin-Watson test for residuals
dw_stats = durbin_watson(results.resid)
print("\nDurbin-Watson Statistics:")
for i, column in enumerate(system_data.columns):
    print(f"{column}: {dw_stats[i]:.4f}")

# Residual Analysis
fig, axes = plt.subplots(3, 1, figsize=(15, 10))
for i, column in enumerate(system_data.columns):
    axes[i].plot(results.resid[column])
    axes[i].set_title(f'Residuals for {column}')
    axes[i].grid(True)
plt.tight_layout()
plt.show() 