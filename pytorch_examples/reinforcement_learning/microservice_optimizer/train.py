import torch
import numpy as np
from environment.service_env import MicroserviceEnv
from agent.dqn_agent import DQNAgent
from utils.visualization import PerformanceVisualizer
from utils.config import TrainingConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    config = TrainingConfig()
    
    # Create environment and agent
    env = MicroserviceEnv(n_services=config.n_services, max_steps=config.max_steps)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_dim = 9  # Number of discrete actions
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        memory_size=config.memory_size,
        batch_size=config.batch_size
    )
    
    visualizer = PerformanceVisualizer()
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in range(config.n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.max_steps):
            # Flatten state
            flat_state = state.reshape(-1)
            
            # Select and perform action
            action = agent.select_action(flat_state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(
                flat_state,
                action,
                reward,
                next_state.reshape(-1),
                done
            )
            
            # Update metrics
            visualizer.add_metrics(info['services'], reward, action)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Train agent
            loss = agent.update()
            
            if done:
                break
        
        # Update target network
        if episode % config.target_update == 0:
            agent.update_target_network()
        
        # Log progress
        logger.info(
            f"Episode {episode}/{config.n_episodes} - "
            f"Reward: {episode_reward:.2f} - "
            f"Epsilon: {agent.epsilon:.2f}"
        )
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), 'best_model.pth')
        
        # Visualize periodically
        if episode % config.save_interval == 0:
            visualizer.plot_service_metrics()
            visualizer.plot_actions_distribution()

if __name__ == "__main__":
    train() 