import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

class VARVisualizer:
    def __init__(self, data, model_results):
        self.data = data
        self.results = model_results
        self.colors = sns.color_palette("husl", 8)
        plt.style.use('seaborn')

    def plot_system_interactions(self):
        """Visualize system variable interactions."""
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 3)
        
        # Diagonal: Distribution plots
        for i, col in enumerate(self.data.columns):
            ax = fig.add_subplot(gs[i, i])
            sns.kdeplot(data=self.data[col], ax=ax)
            ax.set_title(f'{col} Distribution')
        
        # Off-diagonal: Scatter plots with regression
        for i, col1 in enumerate(self.data.columns):
            for j, col2 in enumerate(self.data.columns):
                if i < j:
                    ax = fig.add_subplot(gs[i, j])
                    sns.regplot(data=self.data, x=col1, y=col2, ax=ax)
                elif i > j:
                    ax = fig.add_subplot(gs[i, j])
                    sns.heatmap(
                        pd.DataFrame(
                            np.corrcoef(self.data[col1], self.data[col2])
                        ),
                        annot=True,
                        ax=ax
                    )
        
        plt.tight_layout()
        return fig

    def plot_forecast_evaluation(self, forecast_df):
        """Advanced forecast visualization with uncertainty."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for i, col in enumerate(self.data.columns):
            # Time series and forecast
            axes[i, 0].plot(self.data.index[-100:], 
                          self.data[col][-100:], 
                          label='Observed')
            axes[i, 0].plot(forecast_df.index, 
                          forecast_df[col], 
                          'r--', 
                          label='Forecast')
            
            # Add confidence intervals
            std_err = np.std(self.results.resid[col])
            ci = 1.96 * std_err
            axes[i, 0].fill_between(
                forecast_df.index,
                forecast_df[col] - ci,
                forecast_df[col] + ci,
                color='r',
                alpha=0.2
            )
            
            axes[i, 0].set_title(f'{col} Forecast with 95% CI')
            axes[i, 0].legend()
            
            # Q-Q plot of residuals
            stats.probplot(self.results.resid[col], dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'{col} Residual Q-Q Plot')
        
        plt.tight_layout()
        return fig

    def plot_impulse_diagnostics(self):
        """Enhanced impulse response analysis."""
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 3)
        
        # Impulse responses
        irf = self.results.irf(periods=20)
        axes = []
        for i, shock_var in enumerate(self.data.columns):
            for j, response_var in enumerate(self.data.columns):
                ax = fig.add_subplot(gs[i, j])
                irf.plot_impulse(shock_var, response_var, ax=ax)
                ax.set_title(f'{shock_var} → {response_var}')
                axes.append(ax)
        
        # Cumulative responses
        ax_cum = fig.add_subplot(gs[3, :])
        cumsum_irf = np.cumsum(irf.irfs, axis=0)
        for i, var in enumerate(self.data.columns):
            ax_cum.plot(cumsum_irf[:, i, i], 
                       label=f'{var} Cumulative',
                       color=self.colors[i])
        ax_cum.set_title('Cumulative Impulse Responses')
        ax_cum.legend()
        
        plt.tight_layout()
        return fig 