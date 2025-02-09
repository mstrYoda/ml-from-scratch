import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Create independent variables
X = np.random.normal(size=(n_samples, 3))
X = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])

# Create dependent variable with some noise
y = 2 * X['feature1'] + 0.5 * X['feature2'] - 1 * X['feature3'] + np.random.normal(0, 0.5, n_samples)

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const)
results = model.fit()

# Print summary
print("\nModel Summary:")
print(results.summary())

# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                   for i in range(X_with_const.shape[1])]
print("\nVariance Inflation Factors:")
print(vif_data)

# Diagnostic Plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Regression Diagnostics')

# Residuals vs Fitted
axes[0, 0].scatter(results.fittedvalues, results.resid)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# QQ Plot
sm.graphics.qqplot(results.resid, line='45', fit=True, ax=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Scale-Location
axes[1, 0].scatter(results.fittedvalues, np.sqrt(np.abs(results.resid)))
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Sqrt(|Residuals|)')
axes[1, 0].set_title('Scale-Location')

# Leverage Plot
sm.graphics.influence_plot(results, ax=axes[1, 1])
axes[1, 1].set_title('Influence Plot')

plt.tight_layout()
plt.show() 