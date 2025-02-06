import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict
import numpy as np

class PerformanceVisualizer:
    def __init__(self):
        self.metrics_history = []
        self.rewards_history = []
        self.actions_history = []
    
    def add_metrics(self, metrics: Dict, reward: float, action: np.ndarray):
        self.metrics_history.append(metrics)
        self.rewards_history.append(reward)
        self.actions_history.append(action)
    
    def plot_service_metrics(self):
        """Plot service performance metrics over time."""
        metrics_df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Service Performance Metrics Over Time')
        
        # CPU Usage
        axes[0, 0].plot(metrics_df['cpu_usage'])
        axes[0, 0].set_title('CPU Usage')
        axes[0, 0].set_ylabel('Usage %')
        
        # Memory Usage
        axes[0, 1].plot(metrics_df['memory_usage'])
        axes[0, 1].set_title('Memory Usage')
        axes[0, 1].set_ylabel('Usage %')
        
        # Latency
        axes[1, 0].plot(metrics_df['latency'])
        axes[1, 0].set_title('Latency')
        axes[1, 0].set_ylabel('ms')
        
        # Request Rate
        axes[1, 1].plot(metrics_df['request_rate'])
        axes[1, 1].set_title('Request Rate')
        axes[1, 1].set_ylabel('requests/s')
        
        # Error Rate
        axes[2, 0].plot(metrics_df['error_rate'])
        axes[2, 0].set_title('Error Rate')
        axes[2, 0].set_ylabel('errors/s')
        
        # Rewards
        axes[2, 1].plot(self.rewards_history)
        axes[2, 1].set_title('Rewards')
        axes[2, 1].set_ylabel('reward')
        
        plt.tight_layout()
        plt.show()
    
    def plot_actions_distribution(self):
        """Plot distribution of actions taken by the agent."""
        actions_df = pd.DataFrame(self.actions_history)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=actions_df)
        plt.title('Distribution of Actions')
        plt.ylabel('Scaling Factor')
        plt.xlabel('Resource Type')
        plt.show() 