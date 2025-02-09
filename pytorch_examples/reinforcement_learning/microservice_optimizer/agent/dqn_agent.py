import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        if random.random() < self.epsilon:
            # Exploration: random action
            return np.random.uniform(0.5, 2.0, size=(self.action_dim,))
        
        # Exploitation: select best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
            return self._discrete_to_continuous_action(action)
    
    def _discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        # Convert discrete action index to continuous scaling factors
        # This is a simplified version; you might want to implement more sophisticated mapping
        base_actions = np.array([
            [0.5, 0.5], [0.5, 1.0], [0.5, 2.0],
            [1.0, 0.5], [1.0, 1.0], [1.0, 2.0],
            [2.0, 0.5], [2.0, 1.0], [2.0, 2.0]
        ])
        return base_actions[action_idx]
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 