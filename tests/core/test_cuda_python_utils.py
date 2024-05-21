from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported,
    my_torch_cond,
    my_torch_while_loop
)

import pytest
import torch

# custom "add" function with autograd
class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = input1 + input2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = grad_input2 = grad_output
        return grad_input1, grad_input2

def custom_add(input1, input2):
    return AddFunction.apply(input1, input2)

# custom "mul" function with autograd
class MulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = input1 * input2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = grad_input2 = grad_output
        return grad_input1, grad_input2

def custom_mul(input1, input2):
    return MulFunction.apply(input1, input2)
# -----------------------------------------------

def test_my_torch_cond():
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    a = torch.tensor([1.0], device="cuda")
    b = torch.tensor([2.0], device="cuda")

    pred = torch.tensor(True, device="cuda")

    # unit test for if (not in stream capture -> for original use)
    # c = my_torch_cond(pred, torch.add, torch.mul, [a, b]) # this is the original code, works fine
    c = torch.cond(pred, torch.add, torch.mul, [a, b]) # use torch.cond + torch.add/torch.mul, cause error
    # c = torch.cond(pred, custom_add, custom_mul, [a, b]) # use custom function for add/mul, works fine

    # unit test for else (in stream capture -> additional function for cuda graphs)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    stream_for_graph = torch.cuda.Stream(a.device)
    with torch.cuda.stream(stream_for_graph), torch.inference_mode(), torch.cuda.graph(
        graph, stream=stream_for_graph
    ):
        c_graph = my_torch_cond(pred, torch.add, torch.mul, [a, b])

    # graph.debug_dump("graph.dot")

    # test pred = True
    with torch.inference_mode():
        graph.replay()
        print(c, c_graph)
        assert torch.all(c == c_graph)
        
    # test pred = False
    torch.logical_not(pred)
    c = my_torch_cond(pred, torch.add, torch.mul, [a, b])
    with torch.inference_mode():
        graph.replay()
        print(c, c_graph)
        assert torch.all(c == c_graph)


# test to replace the "cond_op_dense" function with our "my_torch_cond" implementation
from torch._C import DispatchKey
from torch._higher_order_ops.cond import cond_op
cond_op.py_kernels[DispatchKey.CompositeExplicitAutograd] = my_torch_cond

def test_my_torch_cond_replace():
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    a = torch.tensor([1.0], device="cuda")
    b = torch.tensor([2.0], device="cuda")

    pred = torch.tensor(True, device="cuda")

    # unit test for if (not in stream capture -> for original use)
    c = torch.cond(pred, custom_add, custom_mul, [a, b])

    # unit test for else (in stream capture -> additional function for cuda graphs)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    stream_for_graph = torch.cuda.Stream(a.device)
    with torch.cuda.stream(stream_for_graph), torch.inference_mode(), torch.cuda.graph(
        graph, stream=stream_for_graph
    ):
        c_graph = torch.cond(pred, custom_add, custom_mul, [a, b])

    # graph.debug_dump("graph.dot")

    # test pred = True
    with torch.inference_mode():
        graph.replay()
        print(c, c_graph)
        assert torch.all(c == c_graph)
        
    # test pred = False
    torch.logical_not(pred)
    c = torch.cond(pred, custom_add, custom_mul, [a, b])
    with torch.inference_mode():
        graph.replay()
        print(c, c_graph)
        assert torch.all(c == c_graph)


from torch._higher_order_ops.while_loop import while_loop

def test_my_torch_while_loop():
    # imperative programming
    buf_1 = torch.zeros(10, device="cuda")
    for i in range(10):
        buf_1[i] = 10 * i
    
    # pytorch while_loop function
    buf_2 = torch.zeros(10, device="cuda")
    it = torch.tensor(10, device="cuda")
    def cond_fn(i, x):
        return i > 0

    def body_fn(i, x):
        x[i - 1] = 10 * (i - 1)
        return i - 1, x
    
    _, _ = while_loop(cond_fn, body_fn, (it, buf_2))

    import nvtx
    pr = nvtx.Profile()
    torch.cuda.cudart().cudaProfilerStart()
    pr.enable()  # begin annotating function calls
    _, buf_2 = while_loop(cond_fn, body_fn, (it, buf_2))
    pr.disable()  # stop annotating function calls
    torch.cuda.cudart().cudaProfilerStop()

    # my_torch_while_loop
    buf_3 = torch.zeros(10, device="cuda")
    _, buf_3 = my_torch_while_loop(cond_fn, body_fn, (it, buf_3))

    assert torch.all(buf_1 == buf_2)
    assert torch.all(buf_1 == buf_3)
