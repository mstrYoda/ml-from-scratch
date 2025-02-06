from dataclasses import dataclass

@dataclass
class TrainingConfig:
    n_services: int = 3
    n_episodes: int = 1000
    max_steps: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    target_update: int = 10
    save_interval: int = 100

@dataclass
class ServiceConfig:
    min_cpu: float = 0.1
    max_cpu: float = 1.0
    min_memory: float = 0.1
    max_memory: float = 1.0
    min_latency: float = 10
    max_latency: float = 1000
    min_request_rate: float = 1
    max_request_rate: float = 100 