import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)

# Create trend, seasonality, and noise
trend = np.linspace(0, 10, n_samples)
seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
noise = np.random.normal(0, 1, n_samples)

# Combine components
y = trend + seasonality + noise
ts = pd.Series(y, index=dates)

# Decomposition
decomposition = seasonal_decompose(ts, period=365)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(15, 12))
fig.suptitle('Time Series Decomposition')

decomposition.observed.plot(ax=axes[0])
axes[0].set_title('Original Time Series')

decomposition.trend.plot(ax=axes[1])
axes[1].set_title('Trend')

decomposition.seasonal.plot(ax=axes[2])
axes[2].set_title('Seasonal')

decomposition.resid.plot(ax=axes[3])
axes[3].set_title('Residual')

plt.tight_layout()
plt.show()

# Stationarity Test (ADF Test)
adf_result = adfuller(ts)
print('\nAugmented Dickey-Fuller Test:')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

# ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
plot_acf(ts, lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function')

plot_pacf(ts, lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function')

plt.tight_layout()
plt.show()

# Fit SARIMA model
model = sm.tsa.statespace.SARIMAX(
    ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
results = model.fit()

print('\nModel Summary:')
print(results.summary())

# Forecast
forecast = results.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(15, 6))
ts.plot(label='Observed')
forecast_mean.plot(label='Forecast', style='r--')
plt.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='r',
    alpha=0.1
)
plt.title('Time Series Forecast')
plt.legend()
plt.show() 