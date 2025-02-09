import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from scipy import stats

class SurvivalVisualizer:
    def __init__(self, data):
        self.data = data
        self.colors = sns.color_palette("husl", 8)
        plt.style.use('seaborn')

    def plot_survival_distribution(self):
        """Plot survival distribution with confidence bands and risk table."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        kmf = KaplanMeierFitter()
        
        for i, component in enumerate(['A', 'B']):
            mask = self.data['component'] == component
            kmf.fit(
                self.data[mask]['time'],
                ~self.data[mask]['censored'],
                label=f'Component {component}'
            )
            
            # Plot survival curve
            kmf.plot_survival_function(ax=ax1, ci_show=True, color=self.colors[i])
            
            # Add risk table
            at_risk = kmf.survival_function_.reset_index()
            ax2.plot(at_risk['timeline'], 
                    kmf.event_table['at_risk'],
                    drawstyle='steps-post',
                    color=self.colors[i],
                    label=f'Component {component}')
        
        ax1.grid(True)
        ax1.set_title('Survival Distribution with Confidence Intervals')
        ax1.set_xlabel('')
        
        ax2.grid(True)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('At Risk')
        ax2.set_title('Number at Risk')
        
        plt.tight_layout()
        return fig

    def plot_hazard_comparison(self):
        """Create comparative hazard plots with confidence bands."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cumulative hazard
        for i, component in enumerate(['A', 'B']):
            mask = self.data['component'] == component
            kmf = KaplanMeifierFitter()
            kmf.fit(
                self.data[mask]['time'],
                ~self.data[mask]['censored'],
                label=f'Component {component}'
            )
            kmf.plot_cumulative_hazard(ax=axes[0, 0], ci_show=True)
        
        axes[0, 0].set_title('Cumulative Hazard')
        axes[0, 0].grid(True)
        
        # Hazard ratio
        time_grid = np.linspace(0, self.data['time'].max(), 100)
        hazard_ratio = []
        for t in time_grid:
            mask_a = (self.data['component'] == 'A') & (self.data['time'] >= t)
            mask_b = (self.data['component'] == 'B') & (self.data['time'] >= t)
            if sum(mask_a) > 0 and sum(mask_b) > 0:
                ratio = sum(mask_a) / sum(mask_b)
                hazard_ratio.append(ratio)
            else:
                hazard_ratio.append(np.nan)
        
        axes[0, 1].plot(time_grid, hazard_ratio)
        axes[0, 1].set_title('Hazard Ratio (A/B)')
        axes[0, 1].grid(True)
        
        # Log-log plot
        for i, component in enumerate(['A', 'B']):
            mask = self.data['component'] == component
            kmf = KaplanMeifierFitter()
            kmf.fit(
                self.data[mask]['time'],
                ~self.data[mask]['censored'],
                label=f'Component {component}'
            )
            kmf.plot_log_log_survival(ax=axes[1, 0])
        
        axes[1, 0].set_title('Log-Log Survival')
        axes[1, 0].grid(True)
        
        # Conditional survival
        for i, component in enumerate(['A', 'B']):
            mask = self.data['component'] == component
            survival_times = self.data[mask & ~self.data['censored']]['time']
            conditional_survival = []
            for t in time_grid:
                cond_surv = sum(survival_times >= t + 100) / sum(survival_times >= t)
                conditional_survival.append(cond_surv)
            axes[1, 1].plot(time_grid, conditional_survival, 
                          label=f'Component {component}')
        
        axes[1, 1].set_title('100-Hour Conditional Survival')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig 