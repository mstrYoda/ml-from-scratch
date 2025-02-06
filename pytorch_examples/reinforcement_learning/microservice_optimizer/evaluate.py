import torch
import numpy as np
from environment.service_env import MicroserviceEnv
from agent.dqn_agent import DQNAgent
from utils.visualization import PerformanceVisualizer
from utils.config import TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model_path: str, n_episodes: int = 10):
    config = TrainingConfig()
    
    # Create environment and agent
    env = MicroserviceEnv(n_services=config.n_services, max_steps=config.max_steps)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = 9
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_start=0.0  # No exploration during evaluation
    )
    
    # Load trained model
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval()
    
    visualizer = PerformanceVisualizer()
    
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.max_steps):
            # Flatten state
            flat_state = state.reshape(-1)
            
            # Select action
            action = agent.select_action(flat_state)
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            visualizer.add_metrics(info['services'], reward, action)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        logger.info(f"Episode {episode} - Reward: {episode_reward:.2f}")
    
    # Plot results
    visualizer.plot_service_metrics()
    visualizer.plot_actions_distribution()
    
    logger.info(f"Average Reward: {np.mean(total_rewards):.2f}")
    logger.info(f"Std Reward: {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    evaluate('best_model.pth') 