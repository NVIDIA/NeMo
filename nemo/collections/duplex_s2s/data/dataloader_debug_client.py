import os
import socket
import time
from functools import lru_cache

import requests
import torch


@lru_cache(maxsize=1)
def get_dataloader_debug_client(server_url=None):
    if server_url is None:
        return DataloaderDebugClient()
    else:
        return DataloaderDebugClient(server_url)


class DataloaderDebugClient:
    def __init__(self, server_url="http://localhost:5123"):
        """Initialize the metrics client"""
        self.server_url = server_url
        self.hostname = socket.gethostname()

        # For distributed training with PyTorch
        self.rank = int(os.environ.get('RANK', -1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

        # If environment variables aren't set but distributed is initialized
        if self.rank == -1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    @property
    def worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info else 0

    def send_metrics(self, metrics: dict):
        """Send metrics to the server"""
        data = {
            'worker_id': self.worker_id,
            'node_id': self.hostname,
            'gpu_id': self.local_rank if self.local_rank >= 0 else self.rank,
            'metrics': metrics,
            'lhotse_process_seed': int(os.environ.get("LHOTSE_PROCESS_SEED", -1)),
            'timestamp': time.time(),
        }

        try:
            response = requests.post(f"{self.server_url}/metrics", json=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending metrics: {str(e)}")
            return False

    def get_metrics(self):
        """Get all metrics from the server"""
        try:
            response = requests.get(f"{self.server_url}/metrics")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting metrics: {str(e)}")
            return None

    def get_aggregated_metrics(self, agg_type='mean'):
        """Get aggregated metrics from the server"""
        try:
            response = requests.get(f"{self.server_url}/metrics/aggregate?type={agg_type}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting aggregated metrics: {str(e)}")
            return None

    def get_metrics_summary(self):
        """Get a summary of metrics from the server"""
        try:
            response = requests.get(f"{self.server_url}/metrics/summary")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Error getting metrics summary: {str(e)}")
            return None

    def clear_metrics(self):
        """Clear all metrics on the server"""
        try:
            response = requests.post(f"{self.server_url}/metrics/clear")
            return response.status_code == 200
        except Exception as e:
            print(f"Error clearing metrics: {str(e)}")
            return False
