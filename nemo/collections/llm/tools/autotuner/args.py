"""
AutoTuneArgs class for configuration management.

This module contains the AutoTuneArgs class that handles all configuration
parameters and serialization/deserialization for the AutoTuner system.
"""

import os
import json
import base64
import pickle
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AutoTuneArgs:
    """Class to hold all AutoTune arguments and handle serialization."""
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model')
        self.nodes = kwargs.get('nodes')
        self.gpus_per_node = kwargs.get('gpus_per_node')
        self.tensor_parallel_sizes = kwargs.get('tensor_parallel_sizes', [1, 2])
        self.pipeline_parallel_sizes = kwargs.get('pipeline_parallel_sizes', 'auto')
        self.context_parallel_sizes = kwargs.get('context_parallel_sizes', [1, 2])
        self.expert_parallel_sizes = kwargs.get('expert_parallel_sizes', [1])
        self.virtual_pipeline_model_parallel_sizes = kwargs.get('virtual_pipeline_model_parallel_sizes', None)
        self.micro_batch_sizes = kwargs.get('micro_batch_sizes', 'auto')
        self.max_model_parallel_size = kwargs.get('max_model_parallel_size', 8)
        self.min_model_parallel_size = kwargs.get('min_model_parallel_size', 1)
        self.max_steps_per_run = kwargs.get('max_steps_per_run', 10)
        self.max_minutes_per_run = kwargs.get('max_minutes_per_run', 10)
        self.num_tokens_in_b = kwargs.get('num_tokens_in_b', 15000)
        self.vocab_size = kwargs.get('vocab_size', 32000)
        self.seq_length = kwargs.get('seq_length', 8192)
        self.global_batch_sizes = kwargs.get('global_batch_sizes', [512])
        if isinstance(self.global_batch_sizes, tuple):
            self.global_batch_sizes = list(self.global_batch_sizes)
        self.val_check_interval = kwargs.get('val_check_interval', 50)
        self.max_steps = kwargs.get('max_steps', 10)
        self.get_results = kwargs.get('get_results', False)
        self.sequential = kwargs.get('sequential', False)
        # dynamic properties for executor
        self.resource_shape = kwargs.get('resource_shape')
        self.container_image = kwargs.get('container_image', 'nvcr.io/nvidia/nemo:25.04')
        self.nemo_run_dir = kwargs.get('nemo_run_dir', '/nemo-workspace/nemo-run')
        self.mount_path = kwargs.get('mount_path')
        self.mount_from = kwargs.get('mount_from')
        self.launcher_node_group = kwargs.get('launcher_node_group')
        self.training_node_group = kwargs.get('training_node_group')
        self.hf_token = kwargs.get('hf_token', None)
        self.wandb_api_key = kwargs.get('wandb_api_key', None)
        self.torch_home = kwargs.get('torch_home', '/nemo-workspace/.cache')
        self.pythonpath = kwargs.get('pythonpath', '/nemo-workspace/nemo-run:$PYTHONPATH')
        self.memory_per_gpu = kwargs.get('memory_per_gpu')
        self.logs_subdir = kwargs.get('logs_subdir')
        self.config_dir = kwargs.get('config_dir')
        # Metadata from generation results (populated after generate)
        self.metadata = kwargs.get('metadata', {})

    def _serialize_object(self, obj):
        """Serialize a Python object to base64-encoded pickle string."""
        try:
            pickled_data = pickle.dumps(obj)
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            return {
                '_type': 'pickled_object',
                '_class': obj.__class__.__name__,
                '_module': obj.__class__.__module__,
                '_data': encoded_data
            }
        except Exception as e:
            logger.warning(f"Could not serialize object {type(obj).__name__}: {e}")
            return {
                '_type': 'serialization_failed',
                '_class': obj.__class__.__name__,
                '_error': str(e)
            }

    def _deserialize_object(self, obj_dict):
        """Deserialize a base64-encoded pickle string back to Python object."""
        if not isinstance(obj_dict, dict) or obj_dict.get('_type') != 'pickled_object':
            return obj_dict
        try:
            encoded_data = obj_dict['_data']
            pickled_data = base64.b64decode(encoded_data.encode('utf-8'))
            obj = pickle.loads(pickled_data)
            logger.debug(f"Successfully deserialized {obj_dict['_class']} object")
            return obj
        except Exception as e:
            logger.warning(f"Could not deserialize {obj_dict.get('_class', 'unknown')} object: {e}")
            return {
                '_type': 'deserialization_failed',
                '_class': obj_dict.get('_class', 'unknown'),
                '_error': str(e),
                '_original': obj_dict
            }

    def _process_metadata_for_serialization(self, metadata):
        """Process metadata to serialize only non-JSON-serializable objects."""
        processed = {}
        for key, value in metadata.items():
            try:
                # Try to JSON-serialize the value
                json.dumps(value)
                processed[key] = value
            except (TypeError, OverflowError):
                # If not serializable, pickle it
                processed[key] = self._serialize_object(value)
        return processed

    def _process_metadata_for_deserialization(self, metadata):
        """Process metadata to deserialize complex objects."""
        processed = {}
        
        for key, value in metadata.items():
            if isinstance(value, dict) and value.get('_type') == 'pickled_object':
                # Deserialize complex objects
                processed[key] = self._deserialize_object(value)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                processed[key] = self._process_metadata_for_deserialization(value)
            elif isinstance(value, list):
                # Process lists that might contain serialized objects
                processed[key] = [
                    self._deserialize_object(item) if isinstance(item, dict) and item.get('_type') == 'pickled_object'
                    else item for item in value
                ]
            else:
                # Keep simple types as-is
                processed[key] = value
                
        return processed

    def to_dict(self):
        """Convert AutoTuneArgs to dictionary for JSON serialization."""
        processed_metadata = self._process_metadata_for_serialization(self.metadata)
        return {
            'model': self.model,
            'nodes': self.nodes,
            'gpus_per_node': self.gpus_per_node,
            'tensor_parallel_sizes': self.tensor_parallel_sizes,
            'pipeline_parallel_sizes': self.pipeline_parallel_sizes,
            'context_parallel_sizes': self.context_parallel_sizes,
            'expert_parallel_sizes': self.expert_parallel_sizes,
            'virtual_pipeline_model_parallel_sizes': self.virtual_pipeline_model_parallel_sizes,
            'micro_batch_sizes': self.micro_batch_sizes,
            'max_model_parallel_size': self.max_model_parallel_size,
            'min_model_parallel_size': self.min_model_parallel_size,
            'max_steps_per_run': self.max_steps_per_run,
            'max_minutes_per_run': self.max_minutes_per_run,
            'num_tokens_in_b': self.num_tokens_in_b,
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'global_batch_sizes': self.global_batch_sizes,
            'val_check_interval': self.val_check_interval,
            'max_steps': self.max_steps,
            'get_results': self.get_results,
            'sequential': self.sequential,
            'resource_shape': self.resource_shape,
            'container_image': self.container_image,
            'nemo_run_dir': self.nemo_run_dir,
            'mount_path': self.mount_path,
            'mount_from': self.mount_from,
            'launcher_node_group': self.launcher_node_group,
            'training_node_group': self.training_node_group,
            'hf_token': self.hf_token,
            'wandb_api_key': self.wandb_api_key,
            'torch_home': self.torch_home,
            'pythonpath': self.pythonpath,
            'memory_per_gpu': self.memory_per_gpu,
            'logs_subdir': self.logs_subdir,
            'config_dir': self.config_dir,
            'metadata': processed_metadata,
        }

    def get_full_logs_path(self):
        """Get the full logs path by combining mount_path and logs_subdir."""
        return os.path.join(self.mount_path, self.logs_subdir, self.model)
        
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary loaded from JSON."""
        instance = cls(**data)
        if 'metadata' in data:
            instance.metadata = instance._process_metadata_for_deserialization(data['metadata'])
        return instance

    def save_to_file(self, filepath):
        """Save arguments to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filepath):
        """Load arguments from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        instance = cls.from_dict(data)
        
        # infer config_dir from filepath if not set
        if not instance.config_dir:
            # filepath should be: /path/to/config_dir/model/args.json
            # So config_dir should be: /path/to/config_dir
            config_dir = os.path.dirname(os.path.dirname(filepath))
            instance.config_dir = config_dir
            
        return instance

    def update_metadata(self, result):
        """Update metadata with generation results."""
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        
        # Update with all result data (serialization will handle complex objects)
        self.metadata.update(result)

    def update_performance_results(self, performance_dict):
        """Update metadata with performance results."""
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        self.metadata['performance_results'] = performance_dict

    def get_performance_dict(self):
        """Get performance results from metadata."""
        return self.metadata.get('performance_results', {}) if self.metadata else {}

    def has_performance_results(self):
        """Check if performance results exist in metadata."""
        return bool(self.metadata and 'performance_results' in self.metadata)

    def get_memory_analysis(self):
        """Get memory analysis from metadata."""
        return self.metadata.get('memory_analysis', {}) if self.metadata else {}

    def has_memory_analysis(self):
        """Check if memory analysis exists in metadata."""
        return bool(self.metadata and 'memory_analysis' in self.metadata)

    def save_with_metadata(self, filepath, result):
        """Save arguments with updated metadata."""
        self.update_metadata(result)
        self.save_to_file(filepath)

    def get_base_config(self):
        """Get base_config from metadata, with fallback."""
        base_config = self.metadata.get('base_config') if self.metadata else None
        if isinstance(base_config, dict) and base_config.get('_type') == 'deserialization_failed':
            logger.warning("Base config deserialization failed, will need to reconstruct")
            return None
        return base_config

    def get_runner(self):
        """Get runner from metadata, with fallback."""
        runner = self.metadata.get('runner') if self.metadata else None
        if isinstance(runner, dict) and runner.get('_type') == 'deserialization_failed':
            logger.warning("Runner deserialization failed, will need to reconstruct")
            return None
        return runner

    def has_valid_objects(self):
        """Check if we have valid serialized objects."""
        base_config = self.get_base_config()
        runner = self.get_runner()
        return base_config is not None and runner is not None

    def get_configs(self):
        """Get configs from metadata, with fallback."""
        configs = self.metadata.get('configs') if self.metadata else None
        if isinstance(configs, dict):
            # check if any config has deserialization issues
            for key, value in configs.items():
                if isinstance(value, dict) and value.get('_type') == 'deserialization_failed':
                    logger.warning(f"Config '{key}' deserialization failed")
        return configs

    def get_executor_config(self):
        """Get executor configuration for nemo_run."""
        return {
            "resource_shape": self.resource_shape,
            "container_image": self.container_image,
            "nemo_run_dir": self.nemo_run_dir,
            "mount_path": self.mount_path,
            "mount_from": self.mount_from,
            "node_group": self.training_node_group,
            "hf_token": self.hf_token,
            "wandb_api_key": self.wandb_api_key,
            "torch_home": self.torch_home,
            "pythonpath": self.pythonpath,
        } 