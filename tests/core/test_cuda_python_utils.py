from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported,
    my_torch_cond
)

import pytest
import torch

def test_my_torch_cond():
    skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    a = torch.tensor([1.0], device="cuda")
    b = torch.tensor([2.0], device="cuda")

    pred = torch.tensor(True, device="cuda")
    # unit test for if (not in stream capture -> for original use)
    c = my_torch_cond(pred, torch.add, torch.mul, [a, b])

    # unit test for else (in stream capture -> additional function for cuda graphs)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    stream_for_graph = torch.cuda.Stream(a.device)
    with torch.cuda.stream(stream_for_graph), torch.inference_mode(), torch.cuda.graph(
        graph, stream=stream_for_graph
    ):
        c_graph = my_torch_cond(pred, torch.add, torch.mul, [a, b])

    graph.debug_dump("graph.dot")

    print(c, c_graph)

    with torch.inference_mode():
        graph.replay()
        print("hi", c, c_graph)
        assert torch.all(c == c_graph)
