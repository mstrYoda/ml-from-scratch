import numpy as np

# Create date range
dates = np.arange('2024-01', '2024-12', dtype='datetime64[M]')
print("Monthly dates:", dates)

# Working with timedeltas
delta = np.timedelta64(1, 'D')  # One day
dates_daily = np.arange('2024-01-01', '2024-01-10', delta)
print("\nDaily dates:", dates_daily)

# Date arithmetic
dates_plus_week = dates_daily + np.timedelta64(1, 'W')
print("\nDates plus one week:", dates_plus_week)

# Converting between units
hours = np.timedelta64(1, 'h')
print("\nOne hour in different units:")
print("Minutes:", hours / np.timedelta64(1, 'm'))
print("Seconds:", hours / np.timedelta64(1, 's')) 