import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import gym
from gym import spaces

@dataclass
class ServiceMetrics:
    cpu_usage: float
    memory_usage: float
    latency: float
    request_rate: float
    error_rate: float

class MicroserviceEnv(gym.Env):
    """Microservice resource allocation environment."""
    
    def __init__(self, n_services: int = 3, max_steps: int = 100):
        super().__init__()
        self.n_services = n_services
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces
        # Actions: [CPU_scale, Memory_scale] for each service
        self.action_space = spaces.Box(
            low=0.5,  # Minimum scaling factor
            high=2.0,  # Maximum scaling factor
            shape=(n_services, 2)  # (CPU, Memory) for each service
        )
        
        # State: [CPU, Memory, Latency, RequestRate, ErrorRate] for each service
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(n_services, 5)
        )
        
        # Initialize services
        self.services: Dict[int, ServiceMetrics] = {}
        self._initialize_services()

    def _initialize_services(self):
        """Initialize services with random baseline metrics."""
        for i in range(self.n_services):
            self.services[i] = ServiceMetrics(
                cpu_usage=np.random.uniform(0.2, 0.8),
                memory_usage=np.random.uniform(0.2, 0.8),
                latency=np.random.uniform(50, 200),
                request_rate=np.random.uniform(10, 100),
                error_rate=np.random.uniform(0.01, 0.05)
            )

    def _calculate_reward(self) -> float:
        """Calculate reward based on service performance."""
        reward = 0.0
        
        for service in self.services.values():
            # Penalize high resource usage
            resource_penalty = -(service.cpu_usage + service.memory_usage) / 2
            
            # Penalize high latency (normalized)
            latency_penalty = -service.latency / 1000
            
            # Penalize high error rates
            error_penalty = -service.error_rate * 10
            
            # Reward high request handling
            throughput_reward = service.request_rate / 100
            
            reward += resource_penalty + latency_penalty + error_penalty + throughput_reward
        
        return reward

    def _get_state(self) -> np.ndarray:
        """Convert current services state to observation array."""
        state = np.zeros((self.n_services, 5))
        for i, service in self.services.items():
            state[i] = [
                service.cpu_usage,
                service.memory_usage,
                service.latency,
                service.request_rate,
                service.error_rate
            ]
        return state

    def _apply_action(self, action: np.ndarray):
        """Apply scaling actions to services."""
        for i, service in self.services.items():
            cpu_scale, mem_scale = action[i]
            
            # Update resource usage based on scaling
            new_cpu = service.cpu_usage * cpu_scale
            new_memory = service.memory_usage * mem_scale
            
            # Simulate impact on performance
            # More resources generally means better performance
            latency_impact = 1.0 / (cpu_scale * mem_scale)
            error_impact = 1.0 / (cpu_scale * mem_scale)
            
            self.services[i] = ServiceMetrics(
                cpu_usage=np.clip(new_cpu, 0.1, 1.0),
                memory_usage=np.clip(new_memory, 0.1, 1.0),
                latency=np.clip(service.latency * latency_impact, 10, 1000),
                request_rate=service.request_rate,
                error_rate=np.clip(service.error_rate * error_impact, 0.01, 0.5)
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one environment step."""
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'services': self.services
        }
        
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self._initialize_services()
        return self._get_state()

    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print("\nCurrent Environment State:")
            print("-" * 50)
            for i, service in self.services.items():
                print(f"Service {i}:")
                print(f"  CPU Usage: {service.cpu_usage:.2f}")
                print(f"  Memory Usage: {service.memory_usage:.2f}")
                print(f"  Latency: {service.latency:.2f}ms")
                print(f"  Request Rate: {service.request_rate:.2f}/s")
                print(f"  Error Rate: {service.error_rate:.2%}")
                print("-" * 25) 