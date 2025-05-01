import hashlib
import inspect
import json

import torch
import torch.distributed as dist
import torch.nn as nn


def tensor_fingerprint(tensor):
    """
    Create a fingerprint for a tensor that includes:
      - Basic properties: shape, dtype, and device.
      - Statistical summaries: min, max, mean, std, and L2 norm.
      - Sample values from the flattened tensor: first 5, middle 5, and last 5 elements.
      - An MD5 hash of the tensor's raw bytes.
    """
    cpu_tensor = tensor.detach().float().cpu()
    numel = cpu_tensor.numel()

    # Compute statistics (if there is at least one element).
    if numel > 0:
        stats = {
            "min": float(cpu_tensor.min()),
            "max": float(cpu_tensor.max()),
            "mean": float(cpu_tensor.mean()),
            "abs_sum": float(cpu_tensor.abs().sum()),
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "abs_sum": None}

    # Flatten the tensor and extract sample elements.
    # flattened = cpu_tensor.flatten().tolist()
    # if numel == 0:
    #     sample = {}
    # elif numel <= 12:
    #     # If the tensor is small, return all elements.
    #     sample = {"all": flattened}
    # else:
    #     first4 = flattened[:4]
    #     last4 = flattened[-4:]
    #     # Compute a starting index for 4 middle elements.
    #     mid_start = (numel - 4) // 2
    #     middle4 = flattened[mid_start:mid_start+4]
    #     sample = {"first4": first4, "middle4": middle4, "last4": last4}

    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "device": str(tensor.device),
        **stats,
        # "samples": sample,
    }


def safe_convert(obj):
    """
    Recursively convert objects into JSON-serializable representations.
    Tensors are replaced by their fingerprint.
    """
    if isinstance(obj, torch.Tensor):
        return tensor_fingerprint(obj)
    elif isinstance(obj, (list, tuple)):
        return [safe_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return f"<non-serializable: {type(obj).__name__}>"


def get_forward_arg_names(module, inputs):
    """
    Attempt to extract the forward() method's argument names (excluding 'self').
    If the number of names does not match the number of inputs, fall back to default names.
    """
    try:
        sig = inspect.signature(module.forward)
        arg_names = list(sig.parameters.keys())
        if arg_names and arg_names[0] == "self":
            arg_names = arg_names[1:]
        if len(arg_names) != len(inputs):
            return [f"input{i}" for i in range(len(inputs))]
        return arg_names
    except Exception:
        return [f"input{i}" for i in range(len(inputs))]


def create_forward_hook(module_names):
    """
    Create a forward hook that logs:
      - The module's hierarchical (global) name.
      - A mapping of forward input argument names to their fingerprint.
      - The fingerprint of the output.
    """

    def forward_hook(module, inputs, output):
        global_name = module_names.get(id(module), module.__class__.__name__)
        input_names = get_forward_arg_names(module, inputs)
        input_summary = {name: safe_convert(inp) for name, inp in zip(input_names, inputs)}
        output_summary = safe_convert(output)

        try:
            weight_summary = safe_convert(module.weight)
        except Exception:
            weight_summary = "NONE"

        log_entry = {
            "hook": "forward",
            "module": global_name,
            "inputs": input_summary,
            "output": output_summary,
            "weight": weight_summary,
        }

        # Determine the current GPU rank if in a distributed setup.
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        log_filename = f"debug_nemo_fwd_log_{world_size}_rank_{rank}.jsonl"  # JSON Lines format

        try:
            with open(log_filename, "a") as f:
                f.write(json.dumps(log_entry, indent=2) + "\n")
        except Exception as e:
            with open(log_filename, "a") as f:
                error_entry = {"module": global_name, "error": f"Serialization error: {str(e)}"}
                f.write(json.dumps(error_entry) + "\n")

    return forward_hook


def create_backward_hook(module_names):
    """
    Create a backward hook that logs:
      - The module's hierarchical name.
      - Fingerprints of grad_input and grad_output.
    """

    def backward_hook(module, grad_input, grad_output):
        global_name = module_names.get(id(module), module.__class__.__name__)
        grad_input_summary = safe_convert(grad_input)
        grad_output_summary = safe_convert(grad_output)

        log_entry = {
            "hook": "backward",
            "module": global_name,
            "grad_input": grad_input_summary,
            "grad_output": grad_output_summary,
        }

        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        log_filename = f"debug_nemo_bwd_log_{world_size}_rank_{rank}.jsonl"
        # tensor_file = f"debug_nemo_bwd_log_{world_size}_rank_{rank}_{global_name.replace('.', '_')}.pt"

        try:
            with open(log_filename, "a") as f:
                f.write(json.dumps(log_entry, indent=2) + "\n")
                # torch.save({"grad_input": grad_input, "grad_output": grad_output}, tensor_file)
        except Exception as e:
            with open(log_filename, "a") as f:
                error_entry = {"module": global_name, "error": f"Serialization error: {str(e)}"}
                f.write(json.dumps(error_entry) + "\n")

    return backward_hook


def register_hooks(model):
    """
    Register both forward and backward hooks on every submodule of the model.
    A mapping from module id to hierarchical name (using model.named_modules()) is built
    so that each log entry contains a global name (e.g., "layer1.block.MLP") instead of just the class name.
    """
    # Build mapping: module id -> hierarchical name.
    module_names = {
        id(module): name if name != "" else module.__class__.__name__ for name, module in model.named_modules()
    }

    # Create hook functions that share the module names mapping.
    forward_hook_fn = create_forward_hook(module_names)
    # backward_hook_fn = create_backward_hook(module_names)

    # Register both hooks on each module.
    for module in model.modules():
        module.register_forward_hook(forward_hook_fn)
        # module.register_full_backward_hook(backward_hook_fn)
