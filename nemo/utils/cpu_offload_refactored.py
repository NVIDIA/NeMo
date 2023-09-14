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
        ...         logging.info("On save", tensor)
        ...         return (tensor,)
        ...     
        ...     def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        ...         logging.info("On get", saved_state)
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

class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way. 
    The device-to-host and host-to-device copying happen in the same stream 
    as the computation kernels, thus the copying will block computation. 
    """
    def __init__(self, 
                 num_offload_group, 
                 tensor_need_offloading_checker=(lambda _: True), 
                 debug=False
                 ) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker
        self.debug = debug

        self.groupid_reset()

    def groupid_reset(self):
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies the tensor back to gpu and deletes the cpu tensor
        self.current_group, self.tensor_count_current_group = (0, 0) # will increment whenever `group_commit()` is invoked
        self.tensor_tag_to_state = dict()
    
    def on_group_commit_forward(self):
        if self.debug:
            logging.info(f"on_group_commit_forward current_group: {self.current_group}")
        
        # finishing up with updating current group and tensor count
        self.current_group += 1             # increment
        self.tensor_count_current_group = 0 # reset
    
    def on_group_commit_backward(self):
        self.current_group -= 1
        assert self.current_group >= 0

        if self.debug:
            logging.info(f"on_group_commit_backward current_group: {self.current_group}")

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        cpu_backup = torch.empty(src_tensor.size(), 
                                 dtype=src_tensor.dtype,
                                 layout=src_tensor.layout,
                                 device="cpu",
                                 pin_memory=pin_memory)
        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state
    
    @staticmethod
    def reload(state, non_blocking=None):
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            logging.info("tensor_push", tensor_tag, tensor.shape, type(tensor), 
                         "need_offloading ?", self.tensor_need_offloading_checker(tensor))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor # will be offloaded together after group commit
        return tensor_tag
    
    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        if self.debug:
            logging.info("tensor_pop", tensor_tag)
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor

class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, using more memory because of the buffer. But achieves better performance
    due to the overlapping. D2h and h2d copying are completely hidden behind computation if computation time
    of a layer is longer than host-device communication time. Bulk offloading with delay and bulk reloading 
    with prefetch are implemented. """
    def __init__(self, 
                 num_offload_group,     # must be <= actual number of groups (number of commits)
                 num_prefetch_group=1, 
                 tensor_need_offloading_checker=(lambda t: True),
                 debug=False
                 ) -> None:
        super().__init__(num_offload_group=num_offload_group, 
                         tensor_need_offloading_checker=tensor_need_offloading_checker, 
                         debug=debug)
        self.num_prefetch_group = num_prefetch_group
        
        # prepare for tensor buffer
        self.tensor_id_to_tensor_buf_double_bufs = []
        for _ in range(2):
            self.tensor_id_to_tensor_buf_double_bufs.append(dict())

        # allocate streams and events for synchronization
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.h2d_finish_events = []
        self.compute_stream_bwd_start_events = []
        for _ in range(self.num_offload_group):
            self.h2d_finish_events.append(torch.cuda.Event())
            self.compute_stream_bwd_start_events.append(torch.cuda.Event())
        self.d2h_final_event = torch.cuda.Event()

    def get_tensor_buf_for_offloaded_tensor(self, tensor, tensor_tag):
        group_id, tensor_id = tensor_tag
        # obtain ping-pong buffer
        id_buf_map = self.tensor_id_to_tensor_buf_double_bufs[(group_id % 2)]

        if not tensor_id in id_buf_map:
            allocate_new_buf = True
        else:
            tensor_buf = id_buf_map[tensor_id]
            if not (tensor_buf.size() == tensor.size() and tensor_buf.dtype == tensor.dtype):
                allocate_new_buf = True
            else:
                allocate_new_buf = False # in this case, reuse the old buffer

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

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        if self.debug:
            logging.info("tensor_push", tensor_tag, tensor.shape, type(tensor), "need_offloading ?", self.tensor_need_offloading_checker(tensor))
        self.tensor_count_current_group += 1
        assert not (tensor_tag in self.tensor_tag_to_state)
        
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            # first copy the tensor to tensorbuf, so that the original tensor will not be deleted
            tensor_buf = self.get_tensor_buf_for_offloaded_tensor(tensor, tensor_tag)
            tensor_buf.copy_(tensor)
            # Here we just save it, and at commit, bulk_offload_group will handle it
            self.tensor_tag_to_state[tensor_tag] = tensor_buf 
        else:
            self.tensor_tag_to_state[tensor_tag] = tensor
        return tensor_tag
    
    def tensor_pop(self, tensor_tag, **kwargs):
        assert tensor_tag in self.tensor_tag_to_state
        if self.debug:
            logging.info("tensor_pop", tensor_tag)
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        # the tensor should have been copied back in on_group_commit_backward() which invokes bulk_reload_group
        assert not isinstance(tensor, tuple) 
        return tensor  

    def bulk_offload_group(self, group_to_offload):
        with torch.cuda.stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    tensor_on_device = state
                    
                    # if offload, return the reference to cpu copy
                    if self.tensor_need_offloading_checker(tensor_on_device):
                        state = SynchronizedGroupOffloadHandler.offload(tensor_on_device)
                        self.tensor_tag_to_state[tensor_tag] = state

    def synchronize_on_group_commit_forward(self, current_group):
        # the host should wait for the copying of previous group
        # to avoid overwriting buffer
        previous_group = current_group - 1
        if previous_group >= 0 and previous_group < self.num_offload_group:
            torch.cuda.synchronize()
            # TODO (guyueh): this part is originally designed to reduce the peak memory usage.
            # however, uncommenting this part will cause illegal access, have not figured out why.
            
            # if previous_group + 2 >= self.num_offload_group:
            #     # this buffer is no longer required
            #     self.tensor_id_to_tensor_buf_double_bufs[(previous_group % 2)] = dict()

        # the copying of this group should wait for the computation stream event
        if current_group < self.num_offload_group:
            # perform bulk offloading
            self.bulk_offload_group(current_group)
            if current_group == self.num_offload_group - 1:
                self.d2h_stream.record_event(self.d2h_final_event)

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)
        
        # during forward, the next_group_to_fetch always points to the min of 
        # the last commited group, and the last offloaded group
        self.next_group_to_fetch = min(self.current_group, self.num_offload_group -1)

        super().on_group_commit_forward()

    def bulk_reload_group(self, group_to_reload):
        assert group_to_reload < self.num_offload_group
        if group_to_reload == self.num_offload_group - 1:
            self.h2d_stream.wait_event(self.d2h_final_event)
        with torch.cuda.stream(self.h2d_stream):
            # move back tensors
            for tensor_label in self.tensor_tag_to_state.keys():
                group_id, _ = tensor_label
                if group_id == group_to_reload:
                    state = self.tensor_tag_to_state[tensor_label]
                    if isinstance(state, tuple):
                        recovered_tensor = SynchronizedGroupOffloadHandler.reload(state)
                        self.tensor_tag_to_state[tensor_label] = recovered_tensor
                    else:
                        self.tensor_tag_to_state[tensor_label] = state

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1. Finally it should be decremented to 0
        self.current_group -= 1
        assert self.current_group >= 0

        if self.debug:
            logging.info(f"on_group_commit_backward current_group: {self.current_group}")

        # decide the range of group to prefetch
        should_prefetch_until_group = self.current_group - self.num_prefetch_group
        if should_prefetch_until_group < 0:
            should_prefetch_until_group = 0
        
        # do prefetch
        if self.debug:
            logging.info(f"num_prefetch_group = {self.num_prefetch_group} num_offload_group = {self.num_offload_group} fetch from {self.next_group_to_fetch} to {should_prefetch_until_group}")
        for group_num_to_prefetch in range(self.next_group_to_fetch, should_prefetch_until_group - 1, -1):
            # record the event in the compute stream, for h2d to wait
            torch.cuda.current_stream().record_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            # start of h2d should wait for the compute and the d2h
            self.h2d_stream.wait_event(self.compute_stream_bwd_start_events[group_num_to_prefetch])
            
            #recover tensors (copy back from host)
            self.bulk_reload_group(group_num_to_prefetch)
            
            # record an event for the backward of this layer to wait
            self.h2d_stream.record_event(self.h2d_finish_events[group_num_to_prefetch])
        
        self.next_group_to_fetch = min(self.num_offload_group - 1, should_prefetch_until_group - 1) # always is set to -1 at the end of the backward
        
        # wait for the current group
        if self.current_group < self.num_offload_group:
            torch.cuda.current_stream().wait_event(self.h2d_finish_events[self.current_group])

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