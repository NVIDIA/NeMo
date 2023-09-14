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

def dummy_pack(tensor):
    # print("Pack", tensor.shape)
    # print("hasattr main_grad", hasattr(tensor, 'main_grad'))
    return tensor
def dummy_unpack(tensor):
    # print("Unpack", tensor.shape)
    # print("hasattr main_grad", hasattr(tensor, 'main_grad'))
    return tensor

class DummyOffloadHook(CpuOffloadSavedTensorHook):
    def __init__(self) -> None:
        super().__init__()
    
    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        return tensor
    
    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        return saved_state
    
class JitCpuOffloadHook(CpuOffloadSavedTensorHook):
    """Context-manager that just-in-time offloads/recovers a tensor. 
    
    To use this context, you can either directly warp your torch function in this context, 
    or you can use the function `jit_cpu_offload_saved_tensor`. 

    Example:
        x = torch.linear(x, weight)
        with JitCpuOffloadHook():
            y = torch.nn.relu(x)
        y.sum().backward()
    """
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
    """Contex-manager that offloads/recovers tensors through an offload hander.
    
    The hook just offloads/recovers the tensor object to the handler through `tensor_push` and `tensor_pop` interface. 
    How the offload-handler manages the offloading, recovering or prefetching timing is transparent to this hook. 
    """
    def __init__(self, offload_handler, handler_extra_kwargs={}, debug=False) -> None:
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
    """A base class for CPU offload-handler defining two methods."""
    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        raise NotImplementedError("`tensor_push is not implented in OffloadHandler class. "
                                  "Inherit this class and implement your custom tensor_push.")
    
    def tensor_pop(self, state: Any, **kwargs):
        raise NotImplementedError("`tensor_pop is not implented in OffloadHandler class. "
                                  "Inherit this class and implement your custom tensor_pop.")

def offload_saved_tensor_with_handler(
    fwd_func,
    fwd_inputs, 
    offload_handler,
    debug=False
):
    with CpuOffloadHookWithOffloadHandler(offload_handler, debug=debug):
        ret = fwd_func(*fwd_inputs)
    return ret

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

class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to accomplish all synchronizations.
    Implementing it as a function is necessary because we need to actions in both forward and backward.
    """
    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None

group_prefetch_offload_commit = GroupCommitFunction.apply

class GroupAsyncOffloadHandler(OffloadHandler):
    def __init__(self, 
                 num_offload_group, 
                 num_prefetch_group=1, 
                 tensor_need_offloading_checker=(lambda t: True),
                 debug=False
                 ) -> None:
        super().__init__()
        
        self.num_offload_group = num_offload_group
        self.num_prefetch_group = num_prefetch_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.reset()

        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_finish_events = []
        self.h2d_finish_events = []
        self.compute_stream_fwd_finish_events = []
        self.compute_stream_bwd_start_events = []
        for _ in range(self.num_offload_group):
            self.d2h_finish_events.append(torch.cuda.Event())
            self.h2d_finish_events.append(torch.cuda.Event())
            self.compute_stream_fwd_finish_events.append(torch.cuda.Event())
            self.compute_stream_bwd_start_events.append(torch.cuda.Event())

    def reset(self):
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies the tensor back to gpu and deletes the cpu tensor
        #
        # In the future, we may support persistent buffer: The tensor_id_to_buffer will save the cpu tensor
        # for a unique id, and every time a new push happens, as long as the size matches, we will reuse the old 
        # cpu tensor; if size does not match, we will del the old cpu tensor and create a new cpu tensor. This can
        # save alloc time and avoid occupying too much pinned memory.
        self.current_group, self.tensor_count_current_group = (0, 0) # will increment whenever `group_commit()` is invoked
        self.tensor_tag_to_state = dict()
        self.next_group_to_fetch = -1

    def bulk_offload_group(self, group_num):
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_num:
                    assert not isinstance(state, tuple)
                    tensor_on_device = state
                    
                    # if offload, return the reference to cpu copy
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        cpu_backup = torch.empty(tensor_on_device.size(), 
                                                    dtype=tensor_on_device.dtype, 
                                                    layout=tensor_on_device.layout, 
                                                    device='cpu', 
                                                    pin_memory=True)
                        cpu_backup.copy_(tensor_on_device, non_blocking=True)
                        state = (tensor_on_device.device, cpu_backup)
                        self.tensor_tag_to_state[tensor_tag] = state

    def bulk_recover_group(self, group_num):
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label in self.tensor_tag_to_state.keys():
                group_id, _ = tensor_label
                if group_id == group_num:
                    state = self.tensor_tag_to_state[tensor_label]
                    if isinstance(state, tuple):
                        device, cpu_backup = self.tensor_tag_to_state[tensor_label]
                        recovered_tensor = cpu_backup.to(device, non_blocking=cpu_backup.is_pinned())
                        self.tensor_tag_to_state[tensor_label] = recovered_tensor
                    else:
                        self.tensor_tag_to_state[tensor_label] = state
    
    def on_group_commit_forward(self):
        """This function and the `on_group_commit_backward` together accomplishes three synchronizations: 
        1. in forward, record an event in the compute stream; in forward, the d2h stream should wait for it
        2. in forward, record an event in d2h. In backward's prefetch, h2d stream needs to wait for it
        3. in backward's prefetch end, record an event in h2d; in backward, default stream (compute) needs to wait for it
        4. in backward's beginning, record an event in default; prefetch should wait for it.
        """
        if self.debug:
            print(f"on_group_commit_forward current_group: {self.current_group}")
        # insert an event in the compute stream for d2h stream to wait on
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().record_event(self.compute_stream_fwd_finish_events[self.current_group])
            # make d2h wait for compute
            self.d2h_stream.wait_event(self.compute_stream_fwd_finish_events[self.current_group])
            
            # offload tensors
            self.bulk_offload_group(self.current_group)

            # insert an event in d2h for backward to sync on
            self.d2h_stream.record_event(self.d2h_finish_events[self.current_group])
        
        # during forward, the next_group_to_fetch always points to the min between the last commited group, and the last offloaded group
        self.next_group_to_fetch = min(self.current_group, self.num_offload_group -1)

        # finishing up with updating current group and tensor count
        self.current_group += 1             # increment
        self.tensor_count_current_group = 0 # reset

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1. Finally it should be decremented to 0
        self.current_group -= 1
        assert self.current_group >= 0

        if self.debug:
            print(f"on_group_commit_backward current_group: {self.current_group}")

        # decide the range of group to prefetch
        should_prefetch_until_group = self.current_group - self.num_prefetch_group
        if should_prefetch_until_group < 0:
            should_prefetch_until_group = 0
        
        # do prefetch
        if self.debug:
            print(f"num_prefetch_group = {self.num_prefetch_group} num_offload_group = {self.num_offload_group} fetch from {self.next_group_to_fetch} to {should_prefetch_until_group}")
        for group_num_to_prefetch in range(self.next_group_to_fetch, should_prefetch_until_group - 1, -1):
            # record the event in the compute stream, for h2d to wait
            torch.cuda.current_stream().record_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(self.d2h_finish_events[group_num_to_prefetch])
            self.h2d_stream.wait_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            #recover tensors (copy back from host)
            self.bulk_recover_group(group_num_to_prefetch)
            
            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(self.h2d_finish_events[group_num_to_prefetch])
        
        self.next_group_to_fetch = min(self.num_offload_group - 1, should_prefetch_until_group - 1) # always is set to -1 at the end of the backward
        
        # wait for the current group
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().wait_event(self.h2d_finish_events[self.current_group])

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            print("tensor_push", tensor_tag, tensor.shape, type(tensor), "need_offloading ?", self.tensor_need_offloading_checker(tensor))
            # print("hasattr main_grad", hasattr(tensor, "main_grad"))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        self.tensor_tag_to_state[tensor_tag] = tensor # will be offloaded together after group commit

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        if self.debug:
            print("tensor_pop", tensor_tag)
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple) # based on our commit sync mechanism, the tensor should have been copied back 
        # if self.debug:
        #     print("hasattr main_grad", hasattr(tensor, "main_grad"))
        return tensor  

class GroupAsyncOffloadImmediateOffloadHandler(OffloadHandler):
    def __init__(self, 
                 num_offload_group, 
                 num_prefetch_group=1, 
                 tensor_need_offloading_checker=(lambda t: True),
                 debug=False
                 ) -> None:
        super().__init__()
        
        self.num_offload_group = num_offload_group
        self.num_prefetch_group = num_prefetch_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.reset()

        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.d2h_finish_events = []
        self.h2d_finish_events = []
        self.compute_stream_bwd_start_events = []
        for _ in range(self.num_offload_group):
            self.d2h_finish_events.append(torch.cuda.Event())
            self.h2d_finish_events.append(torch.cuda.Event())
            self.compute_stream_bwd_start_events.append(torch.cuda.Event())

    def reset(self):
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies the tensor back to gpu and deletes the cpu tensor
        #
        # In the future, we may support persistent buffer: The tensor_id_to_buffer will save the cpu tensor
        # for a unique id, and every time a new push happens, as long as the size matches, we will reuse the old 
        # cpu tensor; if size does not match, we will del the old cpu tensor and create a new cpu tensor. This can
        # save alloc time and avoid occupying too much pinned memory.
        self.current_group, self.tensor_count_current_group = (0, 0) # will increment whenever `group_commit()` is invoked
        self.tensor_tag_to_state = dict()
        self.next_group_to_fetch = -1

    def bulk_recover_group(self, group_num):
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label in self.tensor_tag_to_state.keys():
                group_id, _ = tensor_label
                if group_id == group_num:
                    state = self.tensor_tag_to_state[tensor_label]
                    if isinstance(state, tuple):
                        device, cpu_backup = self.tensor_tag_to_state[tensor_label]
                        recovered_tensor = cpu_backup.to(device, non_blocking=cpu_backup.is_pinned())
                        self.tensor_tag_to_state[tensor_label] = recovered_tensor
                    else:
                        self.tensor_tag_to_state[tensor_label] = state
    
    def on_group_commit_forward(self):
        """This function and the `on_group_commit_backward` together accomplishes three synchronizations: 
        1. in forward, record an event in the compute stream; in forward, the d2h stream should wait for it
        2. in forward, record an event in d2h. In backward's prefetch, h2d stream needs to wait for it
        3. in backward's prefetch end, record an event in h2d; in backward, default stream (compute) needs to wait for it
        4. in backward's beginning, record an event in default; prefetch should wait for it.
        """
        if self.debug:
            print(f"on_group_commit_forward current_group: {self.current_group}")
        # insert an event in the compute stream for d2h stream to wait on
        if self.current_group < self.num_offload_group:
            # insert an event in d2h for backward to sync on
            self.d2h_stream.record_event(self.d2h_finish_events[self.current_group])
        
        # during forward, the next_group_to_fetch always points to the min between the last commited group, and the last offloaded group
        self.next_group_to_fetch = min(self.current_group, self.num_offload_group -1)

        # finishing up with updating current group and tensor count
        self.current_group += 1             # increment
        self.tensor_count_current_group = 0 # reset

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1. Finally it should be decremented to 0
        self.current_group -= 1
        assert self.current_group >= 0

        if self.debug:
            print(f"on_group_commit_backward current_group: {self.current_group}")

        # decide the range of group to prefetch
        should_prefetch_until_group = self.current_group - self.num_prefetch_group
        if should_prefetch_until_group < 0:
            should_prefetch_until_group = 0
        
        # do prefetch
        if self.debug:
            print(f"num_prefetch_group = {self.num_prefetch_group} num_offload_group = {self.num_offload_group} fetch from {self.next_group_to_fetch} to {should_prefetch_until_group}")
        for group_num_to_prefetch in range(self.next_group_to_fetch, should_prefetch_until_group - 1, -1):
            # record the event in the compute stream, for h2d to wait
            torch.cuda.current_stream().record_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(self.d2h_finish_events[group_num_to_prefetch])
            self.h2d_stream.wait_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            #recover tensors (copy back from host)
            self.bulk_recover_group(group_num_to_prefetch)
            
            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(self.h2d_finish_events[group_num_to_prefetch])
        
        self.next_group_to_fetch = min(self.num_offload_group - 1, should_prefetch_until_group - 1) # always is set to -1 at the end of the backward
        
        # wait for the current group
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().wait_event(self.h2d_finish_events[self.current_group])

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            print("tensor_push", tensor_tag, tensor.shape, type(tensor), "need_offloading ?", self.tensor_need_offloading_checker(tensor))
            # print("hasattr main_grad", hasattr(tensor, "main_grad"))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            cpu_backup = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout)
            with torch.cuda.stream(self.d2h_stream):
                cpu_backup.copy_(tensor, non_blocking=True)
            self.tensor_tag_to_state[tensor_tag] = (tensor.device, cpu_backup)
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        if self.debug:
            print("tensor_pop", tensor_tag)
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        assert not isinstance(tensor, tuple) # based on our commit sync mechanism, the tensor should have been copied back 
        # if self.debug:
        #     print("hasattr main_grad", hasattr(tensor, "main_grad"))
        return tensor  


class GroupJitOffloadHandler(OffloadHandler):
    def __init__(self, 
                 num_offload_group, 
                 tensor_need_offloading_checker=lambda t: True, 
                 debug=False
                 ) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.reset()

    def reset(self):
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies the tensor back to gpu and deletes the cpu tensor
        #
        # In the future, we may support persistent buffer: The tensor_id_to_buffer will save the cpu tensor
        # for a unique id, and every time a new push happens, as long as the size matches, we will reuse the old 
        # cpu tensor; if size does not match, we will del the old cpu tensor and create a new cpu tensor. This can
        # save alloc time and avoid occupying too much pinned memory.
        self.current_group, self.tensor_count_current_group = (0, 0) # will increment whenever `group_commit()` is invoked
        self.tensor_tag_to_state = dict()
    
    def on_group_commit_forward(self):
        if self.debug:
            print(f"on_group_commit_forward current_group: {self.current_group}")
        
        # finishing up with updating current group and tensor count
        self.current_group += 1             # increment
        self.tensor_count_current_group = 0 # reset
    
    def on_group_commit_backward(self):
        self.current_group -= 1
        assert self.current_group >= 0

        if self.debug:
            print(f"on_group_commit_backward current_group: {self.current_group}")

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            print("tensor_push", tensor_tag, tensor.shape, type(tensor), "need_offloading ?", self.tensor_need_offloading_checker(tensor))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            cpu_backup = torch.empty(tensor.size(), 
                                        dtype=tensor.dtype,
                                        layout=tensor.layout,
                                        device="cpu",
                                        pin_memory=True)
            cpu_backup.copy_(tensor, non_blocking=True)
            state = (tensor.device, cpu_backup)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor # will be offloaded together after group commit
        return tensor_tag
    
    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        if self.debug:
            print("tensor_pop", tensor_tag)
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            dev, cpu_backup = state
            tensor = cpu_backup.to(dev)
        else:
            tensor = state
        return tensor

class GroupAsyncPersistentBufferOffloadHandler(GroupAsyncOffloadHandler):
    def __init__(self,
                 num_offload_group,
                 num_prefetch_group=1,
                 tensor_need_offloading_checker=(lambda _: True),
                 debug=False) -> None:
        super().__init__(num_offload_group, num_prefetch_group, tensor_need_offloading_checker, debug)

        self.tensor_id_to_tensor_bufs = []
        for _ in range(2):
            self.tensor_id_to_tensor_bufs.append(dict()) # ping-pong buffer

    def get_tensor_buf_for_offloaded_tensor(self, tensor, tensor_tag):
        group_id, tensor_id = tensor_tag
        id_buf_map = self.tensor_id_to_tensor_bufs[ (group_id % 2) ]

        if not tensor_id in id_buf_map:
            allocate_new_buf = True
        else:
            tensor_buf = id_buf_map[tensor_id]
            if not (tensor_buf.size() == tensor.size() and tensor_buf.dtype == tensor.dtype):
                allocate_new_buf = True
            else:
                allocate_new_buf = False
        # if not (tensor_id in id_buf_map):
        #     allocate_new_buf = True
        # else:
        #     tensor_buf = id_buf_map[tensor_id]
        #     assert ( tensor_buf.size() == tensor.size() and tensor_buf.dtype == tensor.dtype), \
        #     "async offloading requires that the offloaded tensors in each group have identical shapes"
        #     allocate_new_buf = False
        
        if allocate_new_buf:
            # supposed to only execute once
            if self.debug:
                logging.info(f"Allocating tensor_buf for group {group_id} tensor {tensor_id} size {tensor.size()}")
            id_buf_map[tensor_id] = torch.empty(tensor.size(),
                                                dtype=tensor.dtype,
                                                layout=tensor.layout,
                                                device=tensor.device,
                                                )
        

        return id_buf_map[tensor_id]
    
    def on_group_commit_forward(self):
        if self.current_group - 1 >= 0 and self.current_group - 1 < self.num_offload_group:
            torch.cuda.current_stream().wait_event(self.d2h_finish_events[self.current_group - 1])
        return super().on_group_commit_forward()
    
    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            print("tensor_push", tensor_tag, tensor.shape, type(tensor), "need_offloading ?", self.tensor_need_offloading_checker(tensor))
            # print("hasattr main_grad", hasattr(tensor, "main_grad"))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            tensor_buf = self.get_tensor_buf_for_offloaded_tensor(tensor, tensor_tag)
            tensor_buf.copy_(tensor)
            self.tensor_tag_to_state[tensor_tag] = tensor_buf
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag


# mimic the unpad/pad functions in flash_attention 
# but without using torch.nonzero and torch.item (both cause host-device synchronization).
# only useful when attention_mask == None
from einops import rearrange, repeat

def dummpy_unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    max_seqlen_in_batch = hidden_states.shape[1]
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (rearrange(hidden_states, 'b s ... -> (b s) ...'), 
            None,
            cu_seqlens,
            max_seqlen_in_batch)

def dummy_pad_input(hidden_states, unused, batch, seqlen):
    return rearrange(hidden_states, '(b s) ... -> b s ...', b=batch)