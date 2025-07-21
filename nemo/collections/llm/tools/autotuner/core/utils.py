"""
AutoTune utility functions

This module contains ALL extraction functions in one place to eliminate duplication:
- Model size, precision, and parameter extraction
- GPU type and memory specifications  
- Configuration parsing from strings, objects, and names
- Hardware resource parsing
"""

import os
import json
import re
import logging
from nemo.collections import llm
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from typing import Dict, Any, Union, List, Tuple, Optional
from rich.console import Console

from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs, get_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

console = Console()

# ===================== PATTERNS & CONSTANTS =====================

class ExtractionPatterns:
    """Centralized regex patterns for all extraction operations."""
    
    # Model size patterns (used across multiple functions)
    MODEL_SIZE_PATTERNS = [
        r'Config(\d+)B',      # Nemotron3Config4B -> 4B
        r'_(\d+)b_',          # _4b_, _70b_, etc.
        r'_(\d+)B_',          # _4B_, _70B_, etc.
        r'(\d+)b_',           # 4b_, 70b_ at start
        r'(\d+)B_',           # 4B_, 70B_ at start
        r'_(\d+)b\d',         # _4b8, _70b8 (followed by digit)
        r'_(\d+)B\d',         # _4B8, _70B8 (followed by digit)
        r'(\d+)B',            # General pattern for XB
        r'(\d+)b',            # lowercase b
    ]
    
    # GPU resource patterns
    GPU_RESOURCE_PATTERNS = [
    r'gpu\.(\d+)x([a-zA-Z0-9\-]+)',         # gpu.8xh200, gpu.4xh100, gpu.2xa100-40gb
    r'gpu\.([a-zA-Z0-9\-]+)\.(\w+)',        # gpu.a10.6xlarge
    r'gpu\.([a-zA-Z0-9\-]+)',               # gpu.a10, gpu.a100-40gb, gpu.h100-sxm
    r'(\d+)x([a-zA-Z0-9\-]+)',              # 8xh200, 4xh100, 2xa100-40gb
    r'(\d+)x?',                             # Just count: 8x, 8
    ]
    
    # Precision patterns
    PRECISION_PATTERNS = [
        r"precision='([^']+)'",        # precision='bf16'
        r'precision=([^,\)]+)',        # precision=bf16
    ]
    
    # Config name parsing patterns (new format: model_8nodes_tp_2_pp_1_cp_1_ep_1_mbs_1_vp_None_seq_8192_gbs_64)
    CONFIG_NAME_PATTERNS = {
        'nodes': r'(\d+)nodes_',
        'tp': r'tp_(\d+)_',
        'pp': r'pp_(\d+)_', 
        'cp': r'cp_(\d+)_',
        'ep': r'ep_(\d+)_',
        'vp': r'vp_(\w+?)_',
        'mbs': r'mbs_(\d+)_',
        'gbs': r'gbs_(\d+)(?:_|$)',
        'seq_length': r'seq_(\d+)_',
    }
    
    # Data parameter patterns
    DATA_PATTERNS = {
        'seq_length': r'seq_length=(\d+)',
        'micro_batch_size': r'micro_batch_size=(\d+)',
        'global_batch_size': r'global_batch_size=(\d+)'
    }
    
    # Trainer parameter patterns
    TRAINER_SIMPLE_PATTERNS = {
        'accelerator': r"accelerator='([^']+)'",
        'devices': r'devices=(\d+)',
        'num_nodes': r'num_nodes=(\d+)',
        'max_steps': r'max_steps=(\d+)',
        'limit_val_batches': r'limit_val_batches=(\d+)',
        'limit_test_batches': r'limit_test_batches=(\d+)',
        'val_check_interval': r'val_check_interval=(\d+)',
        'log_every_n_steps': r'log_every_n_steps=(\d+)',
        'accumulate_grad_batches': r'accumulate_grad_batches=(\d+)',
        'use_distributed_sampler': r'use_distributed_sampler=(True|False)'
    }
    
    # Strategy patterns
    STRATEGY_PATTERNS = {
        'tensor_model_parallel_size': r'tensor_model_parallel_size=(\d+)',
        'pipeline_model_parallel_size': r'pipeline_model_parallel_size=(\d+)',
        'virtual_pipeline_model_parallel_size': r'virtual_pipeline_model_parallel_size=(None|\d+)',
        'context_parallel_size': r'context_parallel_size=(\d+)',
        'sequence_parallel': r'sequence_parallel=(True|False)',
        'expert_model_parallel_size': r'expert_model_parallel_size=(\d+)',
        'pipeline_dtype': r'pipeline_dtype=(None|[^,\)]+)',
        'ckpt_async_save': r'ckpt_async_save=(True|False)',
        'ckpt_parallel_load': r'ckpt_parallel_load=(True|False)',
        'gradient_as_bucket_view': r'gradient_as_bucket_view=(True|False)',
        'ckpt_include_optimizer': r'ckpt_include_optimizer=(True|False)'
    }

GPU_MEMORY_SPECS = {
    "h100": 80,
    "h200": 141,  
    "a100": 80, 
    "v100": 32, 
    "l40s": 48, 
}

# ===================== CORE EXTRACTION FUNCTIONS =====================

def extract_value_with_patterns(text: str, patterns: List[str], convert_type: type = str, default: Any = None) -> Any:
    """Extract value using multiple regex patterns with type conversion."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = match.group(1)
                if convert_type == bool:
                    return value.lower() == 'true'
                elif convert_type == int:
                    return int(value)
                elif convert_type == float:
                    return float(value)
                else:
                    return value
            except (ValueError, IndexError) as e:
                continue
    return default

def extract_model_size_unified(source: Union[str, Dict[str, Any], object], source_type: str = "auto") -> Optional[float]:
    """
    Unified model size extraction from config values directly (no database lookup).
    
    Args:
        source: String (model name, config name), dict (config dict), or live object
        source_type: "model_name", "config_name", "config_dict", "live_object", or "auto"
        
    Returns:
        Model size in billions of parameters, or None if not found
    """
    if source_type == "auto":
        if isinstance(source, dict):
            source_type = "config_dict"
        elif isinstance(source, str):
            # Heuristic: if it has underscores and parallel indicators, it's a config name
            if any(x in source for x in ['tp_', 'pp_', 'nodes']):
                source_type = "config_name"
            else:
                source_type = "model_name"
        else:
            source_type = "live_object"
    
    if source_type == "config_dict" and isinstance(source, dict):
        try:
            # extracting from __arguments__ model string
            model_value = source.get('__arguments__', {}).get('model', '')
            
            # ensuring we have a string for regex operations
            if model_value:
                # convert to string if it's not already a string (handles Config objects)
                model_str = str(model_value) if not isinstance(model_value, str) else model_value
                
                # skip if the string representation is not useful
                if model_str and model_str not in ['Config', '<Config>', 'None', '']:
                    size = extract_value_with_patterns(
                        model_str, ExtractionPatterns.MODEL_SIZE_PATTERNS, float
                    )
                    if size:
                        return size
        except Exception:
            pass
    
    # extract from model name or config name using pattern matching only
    elif source_type in ["model_name", "config_name"] and isinstance(source, str):
        size = extract_value_with_patterns(
            source, ExtractionPatterns.MODEL_SIZE_PATTERNS, float
        )
        if size:
            return size
    
    # extract from live objects
    elif source_type == "live_object" and not isinstance(source, (str, dict)):
        try:
            if hasattr(source, 'model'):
                model_obj = source.model
                model_str = str(model_obj)
                if model_str and model_str not in ['Config', '<Config>', 'None', '']:
                    size = extract_value_with_patterns(
                        model_str, ExtractionPatterns.MODEL_SIZE_PATTERNS, float
                    )
                    if size:
                        return size
                
                # try to extract from object name/type if available
                if hasattr(source, '__class__'):
                    class_name = source.__class__.__name__
                    size = extract_value_with_patterns(
                        class_name, ExtractionPatterns.MODEL_SIZE_PATTERNS, float
                    )
                    if size:
                        return size
                
                # try to convert entire object to string as fallback
                object_str = str(source)
                if object_str and object_str not in ['Config', '<Config>', 'None', '']:
                    size = extract_value_with_patterns(
                        object_str, ExtractionPatterns.MODEL_SIZE_PATTERNS, float
                    )
                    if size:
                        return size
        except Exception:
            pass
    return None

def extract_precision_unified(source: Union[str, Dict[str, Any], object]) -> str:
    """
    Unified precision extraction.
    
    Args:
        source: String, config dict, or live object
        
    Returns:
        Precision string (bf16, fp16, fp32)
    """
    if isinstance(source, dict):
        try:
            trainer_value = source.get('__arguments__', {}).get('trainer', '')
            if trainer_value:
                trainer_str = str(trainer_value) if not isinstance(trainer_value, str) else trainer_value
                if trainer_str and trainer_str not in ['Config', '<Config>', 'None', '']:
                    precision = extract_value_with_patterns(
                        trainer_str, ExtractionPatterns.PRECISION_PATTERNS, str
                    )
                    if precision:
                        if 'bf16' in precision.lower():
                            return 'bf16'
                        elif 'fp16' in precision.lower():
                            return 'fp16'
                        elif 'fp32' in precision.lower():
                            return 'fp32'
        except Exception:
            pass
    
    elif isinstance(source, str):
        precision = extract_value_with_patterns(
            source, ExtractionPatterns.PRECISION_PATTERNS, str
        )
        if precision:
            if 'bf16' in precision.lower():
                return 'bf16'
            elif 'fp16' in precision.lower():
                return 'fp16'
            elif 'fp32' in precision.lower():
                return 'fp32'
    
    # live objects (nemo_run.config.Partial, etc.)
    elif not isinstance(source, (str, dict)) and hasattr(source, '__dict__'):
        try:
            if hasattr(source, 'trainer'):
                trainer_obj = source.trainer
                trainer_str = str(trainer_obj)
                if trainer_str and trainer_str not in ['Config', '<Config>', 'None', '']:
                    precision = extract_value_with_patterns(
                        trainer_str, ExtractionPatterns.PRECISION_PATTERNS, str
                    )
                    if precision:
                        if 'bf16' in precision.lower():
                            return 'bf16'
                        elif 'fp16' in precision.lower():
                            return 'fp16'
                        elif 'fp32' in precision.lower():
                            return 'fp32'
                object_str = str(source)
                if object_str and object_str not in ['Config', '<Config>', 'None', '']:
                    precision = extract_value_with_patterns(
                        object_str, ExtractionPatterns.PRECISION_PATTERNS, str
                    )
                    if precision:
                        if 'bf16' in precision.lower():
                            return 'bf16'
                        elif 'fp16' in precision.lower():
                            return 'fp16'
                        elif 'fp32' in precision.lower():
                            return 'fp32'
        except Exception:
            pass
    
    return 'bf16'

def extract_gpu_specs_unified(resource_shape: str, memory_per_gpu: Optional[float] = None) -> Tuple[str, int, float]:
    """
    Unified GPU specification extraction.
    
    Args:
        resource_shape: Resource shape string like "gpu.8xh200", "gpu.4xh100", etc.
        memory_per_gpu: Optional custom memory per GPU in GB
        
    Returns:
        Tuple of (gpu_type, gpu_count, memory_per_gpu_gb)
    """
    gpu_type = "h100"
    gpu_count = 8
    
    for pattern in ExtractionPatterns.GPU_RESOURCE_PATTERNS:
        match = re.search(pattern, resource_shape.lower())
        if match:
            if len(match.groups()) >= 2:
                gpu_count = int(match.group(1))
                gpu_type = match.group(2).lower()
                break
            elif len(match.groups()) == 1:
                gpu_count = int(match.group(1))
                break
    
    if memory_per_gpu is not None:
        memory_gb = memory_per_gpu
        logger.info(f"Using custom GPU memory: {memory_per_gpu}GB")
    elif gpu_type in GPU_MEMORY_SPECS:
        memory_gb = GPU_MEMORY_SPECS[gpu_type]
    else:
        memory_gb = 80.0
        logger.warning(f"Unknown GPU type '{gpu_type}', defaulting to 80GB")
    
    return gpu_type, gpu_count, memory_gb

def extract_config_values_unified(source: Union[str, Dict[str, Any], object]) -> Dict[str, Any]:
    """
    Simplified unified configuration value extraction.
    
    Args:
        source: Config name string, config dict, or live config object
        
    Returns:
        Dictionary with ALL extracted values
    """
    default_values = {
        'tp': 1, 'pp': 1, 'cp': 1, 'ep': 1, 'vp': None,
        'mbs': 1, 'gbs': 512, 'nodes': 1, 'seq_length': 8192,
        'model_size_b': None, 'precision': 'bf16'
    }
    
    values = default_values.copy()
    
    values['model_size_b'] = extract_model_size_unified(source)
    values['precision'] = extract_precision_unified(source)
    
    if isinstance(source, str):
        return _extract_from_config_name(source, values)
    
    elif isinstance(source, dict):
        return _extract_from_dict(source, values)
    
    else:
        return _extract_from_live_object(source, values)

# ===================== HELPER EXTRACTION FUNCTIONS =====================

def _extract_from_config_name(source: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Extract values from config name string using regex patterns."""
    for key, pattern in ExtractionPatterns.CONFIG_NAME_PATTERNS.items():
        if key == 'vp':
            vp_val = extract_value_with_patterns(source, [pattern], str)
            values['vp'] = None if not vp_val or vp_val.lower() == 'none' else int(vp_val)
        else:
            extracted = extract_value_with_patterns(source, [pattern], int)
            if extracted is not None:
                values[key] = extracted
    
    return values

def _extract_from_dict(source: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
    if any(key in source for key in ['tp', 'pp', 'cp', 'mbs', 'gbs']):
        values.update({k: v for k, v in source.items() if k in values})
        return values
    
    if '__arguments__' not in source:
        return values
    
    arguments = source['__arguments__']
    
    trainer_values = _extract_trainer_values(arguments.get('trainer', ''))
    values.update(trainer_values)
    
    data_values = _extract_data_values(arguments.get('data', ''))
    values.update(data_values)
    
    return values

def _extract_from_live_object(source: object, values: Dict[str, Any]) -> Dict[str, Any]:
    if type(source).__name__ == 'Partial' and hasattr(source, '__arguments__'):
        partial_dict = {'__arguments__': source.__arguments__}
        return _extract_from_dict(partial_dict, values)
    
    try:
        if hasattr(source, 'trainer'):
            trainer = source.trainer
            if hasattr(trainer, 'num_nodes'):
                values['nodes'] = trainer.num_nodes
            
            if hasattr(trainer, 'strategy'):
                strategy = trainer.strategy
                strategy_mapping = {
                    'tensor_model_parallel_size': 'tp',
                    'pipeline_model_parallel_size': 'pp', 
                    'context_parallel_size': 'cp',
                    'expert_model_parallel_size': 'ep',
                    'virtual_pipeline_model_parallel_size': 'vp'
                }
                for attr, key in strategy_mapping.items():
                    if hasattr(strategy, attr):
                        values[key] = getattr(strategy, attr)
        
        if hasattr(source, 'data'):
            data = source.data
            data_mapping = {
                'micro_batch_size': 'mbs',
                'global_batch_size': 'gbs', 
                'seq_length': 'seq_length'
            }
            for attr, key in data_mapping.items():
                if hasattr(data, attr):
                    values[key] = getattr(data, attr)
    
    except Exception:
        pass
    
    return values

def _extract_trainer_values(trainer_obj: Union[str, object]) -> Dict[str, Any]:
    if not trainer_obj:
        return {}
    
    if hasattr(trainer_obj, 'num_nodes') or hasattr(trainer_obj, 'strategy'):
        try:
            result = {}
            if hasattr(trainer_obj, 'num_nodes'):
                result['nodes'] = trainer_obj.num_nodes
            if hasattr(trainer_obj, 'strategy'):
                strategy = trainer_obj.strategy
                if hasattr(strategy, 'tensor_model_parallel_size'):
                    result['tp'] = strategy.tensor_model_parallel_size
                if hasattr(strategy, 'pipeline_model_parallel_size'):
                    result['pp'] = strategy.pipeline_model_parallel_size
                if hasattr(strategy, 'context_parallel_size'):
                    result['cp'] = strategy.context_parallel_size
                if hasattr(strategy, 'expert_model_parallel_size'):
                    result['ep'] = strategy.expert_model_parallel_size
                if hasattr(strategy, 'virtual_pipeline_model_parallel_size'):
                    result['vp'] = strategy.virtual_pipeline_model_parallel_size
            
            return result
        except Exception as e:
            pass
    
    trainer_str = str(trainer_obj) if not isinstance(trainer_obj, str) else trainer_obj
    trainer_params = _extract_trainer_params_optimized(trainer_str)
    strategy = trainer_params.get('strategy', {})
    
    result = {
        'tp': strategy.get('tensor_model_parallel_size', 1),
        'pp': strategy.get('pipeline_model_parallel_size', 1),
        'cp': strategy.get('context_parallel_size', 1),
        'ep': strategy.get('expert_model_parallel_size', 1),
        'vp': strategy.get('virtual_pipeline_model_parallel_size', None),
        'nodes': trainer_params.get('num_nodes', 1)
    }
    return result

def _extract_data_values(data_obj: Union[str, object]) -> Dict[str, Any]:
    if not data_obj:
        return {}
    
    if hasattr(data_obj, 'micro_batch_size') or hasattr(data_obj, 'global_batch_size') or hasattr(data_obj, 'seq_length'):
        try:
            result = {}
            if hasattr(data_obj, 'micro_batch_size'):
                result['mbs'] = data_obj.micro_batch_size
            if hasattr(data_obj, 'global_batch_size'):
                result['gbs'] = data_obj.global_batch_size
            if hasattr(data_obj, 'seq_length'):
                result['seq_length'] = data_obj.seq_length
            
            return result
        except Exception as e:
            pass
    
    data_str = str(data_obj) if not isinstance(data_obj, str) else data_obj
    data_params = _extract_data_params_optimized(data_str)
    
    result = {
        'mbs': data_params.get('micro_batch_size', 1),
        'gbs': data_params.get('global_batch_size', 512),
        'seq_length': data_params.get('seq_length', 8192)
    }
    return result

def _extract_data_params_optimized(data_str: str) -> Dict[str, Any]:
    data_str = data_str.strip()
    if data_str.startswith('"<Config[MockDataModule(') and data_str.endswith(')]>"'):
        data_str = data_str[24:-3]
    elif data_str.startswith('<Config[MockDataModule(') and data_str.endswith(')]>'):
        data_str = data_str[24:-3]
    
    params = {}
    for param, pattern in ExtractionPatterns.DATA_PATTERNS.items():
        value = extract_value_with_patterns(data_str, [pattern], int)
        if value is not None:
            params[param] = value
    
    return params

def _extract_trainer_params_optimized(trainer_str: str) -> Dict[str, Any]:
    trainer_str = trainer_str.strip()
    if trainer_str.startswith('"<Config[Trainer(') and trainer_str.endswith(')]>"'):
        trainer_str = trainer_str[16:-3]
    elif trainer_str.startswith('<Config[Trainer(') and trainer_str.endswith(')]>'):
        trainer_str = trainer_str[16:-3]
    
    params = {}
    
    for param, pattern in ExtractionPatterns.TRAINER_SIMPLE_PATTERNS.items():
        if param in ['devices', 'num_nodes', 'max_steps', 'limit_val_batches', 
                    'limit_test_batches', 'val_check_interval', 'log_every_n_steps', 
                    'accumulate_grad_batches']:
            value = extract_value_with_patterns(trainer_str, [pattern], int)
        elif param == 'use_distributed_sampler':
            value = extract_value_with_patterns(trainer_str, [pattern], bool)
        else:
            value = extract_value_with_patterns(trainer_str, [pattern], str)
        
        if value is not None:
            params[param] = value
    
    strategy_match = re.search(r'strategy=<Config\[MegatronStrategy\((.*?)\)\]>', trainer_str, re.DOTALL)
    if strategy_match:
        strategy_content = strategy_match.group(1)
        strategy_params = {}
        
        for param, pattern in ExtractionPatterns.STRATEGY_PATTERNS.items():
            if param in ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 
                        'context_parallel_size', 'expert_model_parallel_size']:
                value = extract_value_with_patterns(strategy_content, [pattern], int)
            elif param == 'virtual_pipeline_model_parallel_size':
                match = re.search(pattern, strategy_content)
                if match:
                    val = match.group(1)
                    value = None if val == 'None' else int(val)
                else:
                    value = None
            elif 'True' in pattern or 'False' in pattern:
                value = extract_value_with_patterns(strategy_content, [pattern], bool)
            else:
                value = extract_value_with_patterns(strategy_content, [pattern], str)
            
            if value is not None:
                strategy_params[param] = value
        
        params['strategy'] = strategy_params
    
    return params

# ===================== MODEL SUPPORT FUNCTIONS =====================

def get_supported_models() -> List[str]:
    """Get list of supported models from NeMo's llm module."""
    supported_models = []
    try:
        for attr_name in dir(llm):
            if not attr_name.startswith("_"):
                attr = getattr(llm, attr_name)
                if hasattr(attr, "pretrain_recipe"):
                    supported_models.append(attr_name)
    except Exception as e:
        logger.warning(f"Error getting supported models: {e}")
    return sorted(supported_models)

def validate_model_support(model_name: str) -> Tuple[bool, str]:
    """Validate if a model is supported by NeMo."""
    try:
        supported_models = get_supported_models()
        if model_name in supported_models:
            return True, ""
        
        error_msg = (
            f"Model '{model_name}' is not supported.\n"
            f"Supported models: {', '.join(supported_models)}\n"
            f"For the latest list: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py"
        )
        return False, error_msg
    except Exception as e:
        error_msg = f"Error validating model '{model_name}': {e}"
        return False, error_msg

# ===================== UNIFIED EXTRACTION FUNCTION =====================

def extract_all_values(source: Union[str, Dict[str, Any], object]) -> Dict[str, Any]:
    """
    Extract all relevant configuration values, model info, and precision from any source type.
    Returns a dictionary with keys like 'model_size_b', 'precision', etc.
    """
    return extract_config_values_unified(source)

# ===================== VALIDATION FUNCTIONS =====================

def validate_parallelism_settings(tensor_parallel_sizes: List[int], pipeline_parallel_sizes: List[int], 
                                context_parallel_sizes: List[int], expert_parallel_sizes: List[int],
                                nodes: int, gpus_per_node: int) -> Tuple[bool, str]:
    """
    Validate parallelism settings for consistency.
    """
    total_gpus = nodes * gpus_per_node
    
    for tp in tensor_parallel_sizes:
        if tp <= 0 or total_gpus % tp != 0:
            return False, f"Invalid tensor parallel size {tp} for {total_gpus} GPUs"
    
    if isinstance(pipeline_parallel_sizes, list):
        for pp in pipeline_parallel_sizes:
            if pp <= 0 or total_gpus % pp != 0:
                return False, f"Invalid pipeline parallel size {pp} for {total_gpus} GPUs"
    
    for cp in context_parallel_sizes:
        if cp <= 0 or total_gpus % cp != 0:
            return False, f"Invalid context parallel size {cp} for {total_gpus} GPUs"
    
    for ep in expert_parallel_sizes:
        if ep <= 0 or total_gpus % ep != 0:
            return False, f"Invalid expert parallel size {ep} for {total_gpus} GPUs"
    
    return True, ""

def validate_all_configs(args: Any) -> Tuple[bool, str]:
    """
    Validate all configuration parameters.
    """
    # Validate model support
    is_valid, error_msg = validate_model_support(args.model)
    if not is_valid:
        return False, error_msg
    
    # Validate parallelism settings
    is_valid, error_msg = validate_parallelism_settings(
        args.tensor_parallel_sizes, args.pipeline_parallel_sizes,
        args.context_parallel_sizes, args.expert_parallel_sizes,
        args.nodes, args.gpus_per_node
    )
    if not is_valid:
        return False, error_msg
        
    return True, ""

# ===================== UTILITY FUNCTIONS =====================

def compare_configs_simplified(config1: Union[str, Dict[str, Any], object], config2: Union[str, Dict[str, Any], object]) -> bool:
    """
    Simplified config comparison using the ONE unified extraction function.
    """
    values1 = extract_all_values(config1)
    values2 = extract_all_values(config2)
    
    # compare key values (ignoring some metadata)
    key_fields = ['tp', 'pp', 'cp', 'ep', 'vp', 'mbs', 'gbs', 'seq_length']
    
    for field in key_fields:
        if values1.get(field) != values2.get(field):
            return False
    
    return True

def check_config_matches(base_config_path: str, generated_configs_dir: str) -> Tuple[bool, List[str]]:
    """
    Check if base config matches any generated configs.
    """
    try:
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return False, []
    
    if not os.path.exists(generated_configs_dir):
        return False, []
    
    json_files = [f for f in os.listdir(generated_configs_dir) 
                  if f.endswith('.json') and f not in ['base_config.json', 'args.json']]
    
    matching_files = []
    for filename in json_files:
        filepath = os.path.join(generated_configs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                compare_config = json.load(f)
            
            if compare_configs_simplified(base_config, compare_config):
                matching_files.append(filename)
        except Exception:
            continue
    
    return len(matching_files) > 0, matching_files

def create_log_dir_name(model_name: str, config_values: Dict[str, Any]) -> str:
    """
    Create log directory name in the format: 
    llama_70b_8nodes_tp_4_pp_4_cp_2_ep_1_mbs_1_vp_5_seq_8192_gbs_512
    """
    vp = config_values.get('vp', 'None')
    if vp is None:
        vp = 'None'
    
    return (f"{model_name}_{config_values['nodes']}nodes_"
            f"tp_{config_values['tp']}_pp_{config_values['pp']}_"
            f"cp_{config_values['cp']}_ep_{config_values['ep']}_"
            f"mbs_{config_values['mbs']}_vp_{vp}_"
            f"seq_{config_values['seq_length']}_gbs_{config_values['gbs']}")

def get_args_file_path(model, config_dir):
    """Get the standard path for the args file."""
    return os.path.join(config_dir, model, "args.json")

def update_args_with_generation_metadata(model_name, result, config_dir):
    """Update the args.json file with generation metadata."""
    args_file_path = get_args_file_path(model_name, config_dir)
    args = AutoTuneArgs.load_from_file(args_file_path)
    args.save_with_metadata(args_file_path, result)
    return args_file_path

def _load_args_from_config_dir(config_dir, model=None):
    """Load AutoTuneArgs from a config directory, optionally for a specific model."""
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if model is None:
        subdirs = [d for d in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, d))]
        if not subdirs:
            raise FileNotFoundError(f"No model directories found in: {config_dir}")
        model = subdirs[0]
    args_file_path = get_args_file_path(model, config_dir)
    if not os.path.exists(args_file_path):
        raise FileNotFoundError(f"Args file not found: {args_file_path}")
    return AutoTuneArgs.load_from_file(args_file_path)

def update_args_with_performance_results(model_name, performance_dict, config_dir):
    """Update the args.json file with performance results."""
    try:
        args_file_path = get_args_file_path(model_name, config_dir)
        if os.path.exists(args_file_path):
            args = AutoTuneArgs.load_from_file(args_file_path)
            # Update with performance results and save
            args.update_performance_results(performance_dict)
            args.save_to_file(args_file_path)
            logger.info(f"Performance results saved to {args_file_path}")
        else:
            logger.warning(f"Args file not found: {args_file_path}")
    except Exception as e:
        logger.error(f"Failed to update performance results: {e}")
        raise