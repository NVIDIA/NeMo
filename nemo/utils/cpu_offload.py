import torch
from typing import Any
from nemo.utils import logging

class CpuOffloadSavedTensorHook:
    """Contex-manager that executes a pair of pack/unpack hooks for saved tensors.
    
    In this context, the ``on_save_for_backward`` method will be called every time 
    a tensor is saved for backward (this includes intermediary results saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). 

    The ``on_get_saved_tensors`` method will be called when the backward function
    of this op attempts to retrieve the saved tensor from context (this includes 
    :func: `torch.Tensor.backward()` or :func: `torch.autograd.grad()`. It takes the 
    as input the return value of the ``on_save_for_backward``, and is meant to return
    an identical copy of the tensor being saved by ``on_save_for_backward`` in terms of 
    size, device and element values.

    Example:
        
        >>> import torch
        >>> from typing import Any
        >>> 
        >>> class DummyHook(CpuOffloadSavedTensorHook):
        ...     
        ...     def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        ...         print("On save", tensor)
        ...         return (tensor,)
        ...     
        ...     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        ...         print("On get", saved_state)
        ...         tensor, = saved_state
        ...         return tensor
        ... 
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with DummyHook():
        ...     y = a * b
        ... 
        On save tensor([1., 1., 1., 1., 1.], requires_grad=True)
        On save tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        On get (tensor([1., 1., 1., 1., 1.], requires_grad=True),)
        On get (tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>),)

    """

    def __init__(self) -> None:
        pass
    
    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.on_save_for_backward, 
            self.on_get_saved_tensor
            )
    
    def __exit__(self, *args: Any):
        torch._C._autograd._pop_saved_tensors_default_hooks()
    

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        raise NotImplementedError("`on_save_for_backward: Callable[[torch.Tensor], Any]`" 
                                  "is not implemented in CpuOffloadHook class. Inherit "
                                  "this class and implement your custom hooks")

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        raise NotImplementedError("`on_get_saved_tensors: Callable[[Any], torch.Tensor]`" 
                                  "is not implemented in CpuOffloadHook class. Inherit "
                                  "this class and implement your custom hooks")

class JitCpuOffloadHook(CpuOffloadSavedTensorHook):
    def __init__(self, debug=False) -> None:
        self.debug = debug
        super().__init__()
    
    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        cpu_backup = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout, 
                                 device='cpu', pin_memory=True)
        cpu_backup.copy_(tensor, non_blocking=True)

        if self.debug:
            logging.info(f"Backup tensor {tensor} to CPU {cpu_backup}")
        
        return (tensor.device, cpu_backup)
    
    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        target_device, cpu_backup = saved_state
        ret = cpu_backup.to(target_device, non_blocking=cpu_backup.is_pinned())
        
        if self.debug:
            logging.info(f"Move CPU backup {cpu_backup} to device {ret}")
        
        return ret

def jit_cpu_offload_saved_tensor(
        fwd_func,
        fwd_inputs,
        debug=False
):
    """wrapper function for a module forward or torch function forward that 
    offloads any saved tensor immediately to CPU (non-blocking), and moves the 
    cpu tensors back to original device on the time when the tensor is retrieved 
    through ... = ctx.saved_tensors. 

    This function provides very limited overlapping capability.

    Example:
        x = torch.linear(x, weight)
        y = jit_cpu_offload_saved_tensor(torch.nn.relu, [x])
        y.sum().backward()
    """
    with JitCpuOffloadHook(debug=debug):
        ret = fwd_func(*fwd_inputs)
    return ret

class CpuOffloadHookWithOffloadHandler(CpuOffloadSavedTensorHook):
    def __init__(self, offload_handler, handler_extra_kwargs, debug=False) -> None:
        self.debug = debug
        self.offload_handler = offload_handler
        self.handler_extra_kwargs = handler_extra_kwargs
        super().__init__()
    
    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(
            tensor,
            **self.handler_extra_kwargs 
        )
        if self.debug:
            logging.info(f"On save tensor shape {tensor.shape} parameter {type(tensor)}, offload_handler returns identifier {retrieve_identifier}")
        return retrieve_identifier
    
    def on_get_saved_tensor(self, retrieve_identifier: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(
            retrieve_identifier, 
            **self.handler_extra_kwargs
        )
        if self.debug:
            logging.info(f"On get tensor, from identifier {retrieve_identifier} get tensor shape {tensor.shape}")
        return tensor

class OffloadHandler:
    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        raise NotImplementedError("`tensor_push is not implented in OffloadHandler class. "
                                  "Inherit this class and implement your custom tensor_push.")
    
    def tensor_pop(self, state: Any, **kwargs):
        raise NotImplementedError("`tensor_pop is not implented in OffloadHandler class. "
                                  "Inherit this class and implement your custom tensor_pop.")
    
class BucketPrefetchOffloadHandler(OffloadHandler):
    def __init__(self, num_buckets_to_offload, prefetch_num_buckets, debug=False) -> None:
        self.num_buckets_to_offload = num_buckets_to_offload
        self.prefetch_num_buckets = prefetch_num_buckets
        
        self.saved_tensor_states = dict() # uid -> list of state

        self.latest_bwd_bucket = None
        self.next_bucket_to_fetch = None

        self.D2H_stream = torch.cuda.Stream()
        self.H2D_stream = torch.cuda.Stream()
        self.H2D_events = dict()

        self.debug = debug

    def offload(self, tensor: torch.Tensor):
        cpu_backup = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout, 
                                 device='cpu', pin_memory=True)
        with torch.cuda.stream(self.D2H_stream):
            cpu_backup.copy_(tensor, non_blocking=True)
            if self.debug:
                logging.info(f"Copy D2H of tensor shape {tensor.shape}")
        return (tensor.device, cpu_backup)
    

    def retrieve(self, saved_state):
        target_device, cpu_backup = saved_state
        if self.debug:
            logging.info(f"Copy H2D of tensor shape {cpu_backup.shape}")
        return cpu_backup.to(target_device, non_blocking=cpu_backup.is_pinned())
    
    def need_offloading(self, tensor, bucket_id):
        return bucket_id < self.num_buckets_to_offload and (not isinstance(tensor, torch.nn.Parameter))
    
    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        bucket_id = kwargs['bucket_id']

        if self.need_offloading(tensor, bucket_id):
            state = self.offload(tensor)
        else:
            state = tensor
        
        if not (bucket_id in self.saved_tensor_states):
            self.saved_tensor_states[bucket_id] = dict()
        
        tensor_uid = len(self.saved_tensor_states[bucket_id])
        self.saved_tensor_states[bucket_id][tensor_uid] = state

        if bucket_id == 0 and tensor_uid == 0:
            self.latest_bwd_bucket = None # one-time reset of this variable every fwd+bwd
            self.next_bucket_to_fetch = None

        if self.debug:
            logging.info(f"Added bucket {bucket_id} tensor {tensor_uid}")
        return bucket_id, tensor_uid
    
    def tensor_pop(self, uid: Any, **kwargs):
        bucket_id, tensor_uid = uid

        if self.latest_bwd_bucket != bucket_id:
            if self.latest_bwd_bucket != None and self.latest_bwd_bucket != bucket_id + 1:
                raise ValueError(f"Latest bwd bucket id {self.latest_bwd_bucket} should be the same or +1 of this bucket {bucket_id}")
            
            # jobs on the start of bwd of a new bucket
            
            # 1. prefetch
            if self.next_bucket_to_fetch == None:
                fetch_range_start = self.num_buckets_to_offload -1
            else:
                fetch_range_start = self.next_bucket_to_fetch
            fetch_range_end = max(0, bucket_id - self.prefetch_num_buckets)

            for _bucket_id in range(fetch_range_start, fetch_range_end -1, -1):
                self.start_h2d(_bucket_id)
            self.next_bucket_to_fetch = min(fetch_range_end -1, self.num_buckets_to_offload -1)

            # wait on current
            if bucket_id < self.num_buckets_to_offload:
                self.wait_on_h2d(bucket_id)

            # update the latest_bwd_bucket number
            self.latest_bwd_bucket = bucket_id
        
        # retrieve tensor
        tensor = self.saved_tensor_states[bucket_id].pop(tensor_uid)
        assert isinstance(tensor, torch.Tensor)
        return tensor

    def start_h2d(self, bucket_id):
        
        # for the first backward it needs to wait for the last forward D2h transfer
        if bucket_id == self.num_buckets_to_offload -1:
            self.H2D_stream.wait_stream(self.D2H_stream)
            if self.debug:
                logging.info(f"Finish waiting D2H of bucket {self.num_buckets_to_offload -1}")
            
        new_event = torch.cuda.Event()
        self.H2D_events[bucket_id] = new_event
        with torch.cuda.stream(self.H2D_stream):
            if self.debug:
                logging.info(f"Start copying H2D of bucket {bucket_id}")
            for tensor_uid, state in self.saved_tensor_states[bucket_id].items():
                if not isinstance(state, torch.Tensor): # this is offloaded
                    tensor = self.retrieve(state)
                else:
                    tensor = state
                self.saved_tensor_states[bucket_id][tensor_uid] = tensor
            self.H2D_stream.record_event(self.H2D_events[bucket_id])
    
    def wait_on_h2d(self, bucket_id):
        assert bucket_id in self.H2D_events
        torch.cuda.current_stream().wait_event(self.H2D_events[bucket_id])
        if self.debug:
            logging.info(f"Finish waiting on {bucket_id}")

            
def bucket_prefetch_offload_saved_tensor(
    fwd_func,
    fwd_inputs, 
    bucket_id,
    offload_handler,
    debug=False
):
    with CpuOffloadHookWithOffloadHandler(offload_handler, {'bucket_id': bucket_id}, debug):
        ret = fwd_func(*fwd_inputs)
    return ret