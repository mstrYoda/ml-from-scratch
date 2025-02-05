import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = {
    'date': dates,
    'sales': np.random.normal(100, 20, 365) + \
            np.sin(np.linspace(0, 2*np.pi, 365)) * 20,  # Add seasonality
    'temperature': np.random.normal(20, 5, 365)
}
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

print("Time Series Operations:")
# Resampling
print("\nMonthly average sales:")
print(df.resample('ME')['sales'].mean())

# Rolling statistics
print("\nRolling 7-day average sales:")
print(df['sales'].rolling(window=7).mean().head())

# Shifting data
print("\nSales with 1-day lag:")
df['sales_lag1'] = df['sales'].shift(1)
print(df.head())

# Time-based indexing
print("\nJanuary data:")
print(df['2023-01-01':'2023-01-31'].head())

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['sales'], period=30)
print("\nSeasonal decomposition components:")
print("Trend, Seasonal, Residual shapes:", 
      decomposition.trend.shape,
      decomposition.seasonal.shape,
      decomposition.resid.shape)