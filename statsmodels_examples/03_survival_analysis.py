import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, WeibullFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import seaborn as sns

# Generate sample reliability data
np.random.seed(42)

def generate_failure_data(n_samples=200):
    # Generate failure times for two different components/conditions
    component_A = np.random.weibull(shape=2.0, scale=1000, size=n_samples)
    component_B = np.random.weibull(shape=1.5, scale=800, size=n_samples)
    
    # Add censoring (some components didn't fail during observation)
    observation_time = 1000
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': np.concatenate([component_A, component_B]),
        'component': ['A'] * n_samples + ['B'] * n_samples,
        'temperature': np.concatenate([
            np.random.normal(50, 10, n_samples),
            np.random.normal(60, 10, n_samples)
        ]),
        'load': np.concatenate([
            np.random.normal(75, 15, n_samples),
            np.random.normal(80, 15, n_samples)
        ])
    })
    
    # Add censoring
    data['censored'] = data['time'] > observation_time
    data['time'] = np.minimum(data['time'], observation_time)
    
    return data

# Generate data
failure_data = generate_failure_data()

# 1. Non-parametric Analysis (Kaplan-Meier)
kmf = KaplanMeierFitter()

plt.figure(figsize=(12, 6))
for component in ['A', 'B']:
    mask = failure_data['component'] == component
    kmf.fit(
        failure_data[mask]['time'],
        ~failure_data[mask]['censored'],
        label=f'Component {component}'
    )
    kmf.plot()

plt.title('Kaplan-Meier Survival Curves by Component')
plt.xlabel('Time (hours)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

# 2. Parametric Analysis (Weibull)
wf = WeibullFitter()
wf.fit(failure_data['time'], ~failure_data['censored'])

print("\nWeibull Model Parameters:")
print(wf.print_summary())

# Plot Weibull fit
plt.figure(figsize=(12, 6))
wf.plot_survival_function()
plt.title('Weibull Model Fit')
plt.grid(True)
plt.show()

# 3. Cox Proportional Hazards Model
# Prepare data for Cox model
cph = CoxPHFitter()
cph.fit(
    failure_data,
    duration_col='time',
    event_col='censored',
    covariates=['temperature', 'load', 'component']
)

print("\nCox Proportional Hazards Model:")
print(cph.print_summary())

# Plot partial effects
plt.figure(figsize=(15, 5))
for i, covariate in enumerate(['temperature', 'load']):
    plt.subplot(1, 2, i+1)
    cph.plot_partial_effects(covariate, values=[20, 40, 60, 80])
    plt.title(f'Partial Effect of {covariate}')
    plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Reliability Metrics
def calculate_reliability_metrics(group_data):
    kmf = KaplanMeierFitter()
    kmf.fit(group_data['time'], ~group_data['censored'])
    
    metrics = {
        'MTTF': kmf.mean_survival_time_,
        'Reliability_at_500': kmf.survival_function_at_times(500).values[0],
        'Median_Life': kmf.median_survival_time_
    }
    return pd.Series(metrics)

reliability_metrics = failure_data.groupby('component').apply(calculate_reliability_metrics)
print("\nReliability Metrics by Component:")
print(reliability_metrics)

# 5. Hazard Rate Analysis
def plot_hazard_rate(data, component):
    wf = WeibullFitter()
    mask = data['component'] == component
    wf.fit(data[mask]['time'], ~data[mask]['censored'])
    
    return wf.hazard_at_times(np.linspace(0, 1000, 100))

plt.figure(figsize=(12, 6))
times = np.linspace(0, 1000, 100)
for component in ['A', 'B']:
    hazard_rate = plot_hazard_rate(failure_data, component)
    plt.plot(times, hazard_rate, label=f'Component {component}')

plt.title('Hazard Rate Analysis')
plt.xlabel('Time (hours)')
plt.ylabel('Hazard Rate')
plt.legend()
plt.grid(True)
plt.show() 