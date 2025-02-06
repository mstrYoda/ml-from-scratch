import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class ServiceMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    latency: float
    request_rate: float
    error_rate: float

class MetricsCollector:
    def __init__(self, n_services: int):
        self.n_services = n_services
        self.metrics_history: Dict[int, List[ServiceMetrics]] = {
            i: [] for i in range(n_services)
        }
    
    def collect_metrics(self, service_id: int, metrics: ServiceMetrics):
        """Store metrics for a service."""
        self.metrics_history[service_id].append(metrics)
    
    def get_service_history(self, service_id: int) -> List[ServiceMetrics]:
        """Get metrics history for a specific service."""
        return self.metrics_history[service_id]
    
    def get_average_metrics(self, window_size: int = 10) -> Dict[int, ServiceMetrics]:
        """Calculate average metrics over recent window."""
        averages = {}
        for service_id in range(self.n_services):
            history = self.metrics_history[service_id][-window_size:]
            if not history:
                continue
            
            avg_metrics = ServiceMetrics(
                timestamp=time.time(),
                cpu_usage=np.mean([m.cpu_usage for m in history]),
                memory_usage=np.mean([m.memory_usage for m in history]),
                latency=np.mean([m.latency for m in history]),
                request_rate=np.mean([m.request_rate for m in history]),
                error_rate=np.mean([m.error_rate for m in history])
            )
            averages[service_id] = avg_metrics
        
        return averages 