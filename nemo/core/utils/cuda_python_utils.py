# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib

import numpy as np
import torch
from packaging.version import Version
from functools import wraps

__CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__ = (12, 3)  # 12030

try:
    from cuda import cuda, cudart, nvrtc

    HAVE_CUDA_PYTHON = True
except (ImportError, ModuleNotFoundError):
    HAVE_CUDA_PYTHON = False


def cuda_python_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not HAVE_CUDA_PYTHON:
            raise ModuleNotFoundError("No `cuda-python` module. Please do `pip install cuda-python>=12.3`")
        return f(*args, **kwargs)

    return wrapper


def check_cuda_python_cuda_graphs_conditional_nodes_supported():
    try:
        from cuda import cuda
    except ImportError:
        raise ModuleNotFoundError("No `cuda-python` module. Please do `pip install cuda-python>=12.3`")

    from cuda import __version__ as cuda_python_version

    if Version(cuda_python_version) < Version("12.3.0"):
        raise ImportError(f"Found cuda-python {cuda_python_version}, but at least version 12.3.0 is needed.")

    error, driver_version = cuda.cuDriverGetVersion()
    if error != cuda.CUresult.CUDA_SUCCESS:
        raise ImportError(f"cuDriverGetVersion() returned {cuda.cuGetErrorString(error)}")

    driver_version_major = driver_version // 1000
    driver_version_minor = (driver_version % 1000) // 10

    driver_version = (driver_version_major, driver_version_minor)
    if driver_version < __CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__:
        required_version = __CUDA_PYTHON_MINIMUM_VERSION_CUDA_GRAPH_CONDITIONAL_NODES_SUPPORTED__
        raise ImportError(
            f"""Driver supports cuda toolkit version \
{driver_version_major}.{driver_version_minor}, but the driver needs to support \
at least {required_version[0]},{required_version[1]}. Please update your cuda driver."""
        )


def skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported():
    """
    Helper method to skip pytest test case if cuda graph conditionals nodes are not supported.
    """
    try:
        check_cuda_python_cuda_graphs_conditional_nodes_supported()
    except (ImportError, ModuleNotFoundError) as e:
        import pytest

        pytest.skip(
            f"Test using cuda graphs with conditional nodes is being skipped because cuda graphs with conditional nodes aren't supported. Error message: {e}"
        )


def assert_drv(err):
    """
    Throws an exception if the return value of a cuda-python call is not success.
    """
    from cuda import cuda, cudart, nvrtc

    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cu_call(f_call_out):
    """
    Makes calls to cuda-python's functions inside cuda.cuda more python by throwing an exception if they return a status which is not cudaSuccess
    """
    from cuda import cudart

    error, *others = f_call_out
    if error != cudart.cudaError_t.cudaSuccess:
        raise Exception(f"CUDA failure! {error}")
    else:
        return tuple(others)


@contextlib.contextmanager
def with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle, device):
    """
    Even though we add a conditional node only once, we need to
    capture the kernel that calls cudaGraphSetConditional() both
    before in the parent graph containing the while loop body graph
    and after the rest of the while loop body graph (because we need
    to decide both whether to enter the loop, and also whether to
    execute the next iteration of the loop).
    """
    from cuda import __version__ as cuda_python_version
    from cuda import cuda, cudart, nvrtc

    capture_status, _, graph, _, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=device).cuda_stream)
    )
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    cuda.cuLaunchKernel(
        while_loop_kernel,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        torch.cuda.current_stream(device=device).cuda_stream,
        while_loop_args.ctypes.data,
        0,
    )

    capture_status, _, graph, dependencies, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=device).cuda_stream)
    )
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    driver_params = cuda.CUgraphNodeParams()
    driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    driver_params.conditional.handle = while_loop_conditional_handle
    driver_params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
    driver_params.conditional.size = 1
    if Version(cuda_python_version) == Version("12.3.0"):
        # Work around for https://github.com/NVIDIA/cuda-python/issues/55
        # Originally, cuda-python version 12.3.0 failed to allocate phGraph_out
        # on its own.
        # This bug is fixed in cuda-python version 12.4.0. In fact, we can
        # no longer write to phGraph_out in cuda-python 12.4.0, so we must
        # condition on the version number.
        driver_params.conditional.phGraph_out = [cuda.CUgraph()]
    (ctx,) = cu_call(cuda.cuCtxGetCurrent())
    driver_params.conditional.ctx = ctx

    # Use driver API here because of bug in cuda-python runtime API: https://github.com/NVIDIA/cuda-python/issues/55
    # TODO: Change call to this after fix goes in (and we bump minimum cuda-python version to 12.4.0):
    # node, = cu_call(cudart.cudaGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    (node,) = cu_call(cuda.cuGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    body_graph = driver_params.conditional.phGraph_out[0]

    cu_call(
        cudart.cudaStreamUpdateCaptureDependencies(
            torch.cuda.current_stream(device=device).cuda_stream,
            [node],
            1,
            cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
        )
    )
    body_stream = torch.cuda.Stream(device)
    previous_stream = torch.cuda.current_stream(device=device)
    cu_call(
        cudart.cudaStreamBeginCaptureToGraph(
            body_stream.cuda_stream,
            body_graph,
            None,
            None,
            0,
            cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
        )
    )
    torch.cuda.set_stream(body_stream)

    yield body_stream, body_graph

    cuda.cuLaunchKernel(
        while_loop_kernel, 1, 1, 1, 1, 1, 1, 0, body_stream.cuda_stream, while_loop_args.ctypes.data, 0
    )

    cudart.cudaStreamEndCapture(body_stream.cuda_stream)

    torch.cuda.set_stream(previous_stream)


def run_nvrtc(kernel_string: str, kernel_name: bytes, program_name: bytes):
    from cuda import cuda, nvrtc

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), program_name, 0, [], [])
    assert_drv(err)
    # Compile program
    # Not specifying --gpu-architecture will default us to a fairly low compute capability, which is a safe bet.
    # Otherwise, there are ways to query the current device's compute capability.
    # https://stackoverflow.com/questions/48283009/nvcc-get-device-compute-capability-in-runtime
    opts = []
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert_drv(err)
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    assert_drv(err)
    buf = b" " * size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)
    assert_drv(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    assert_drv(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    assert_drv(err)
    err, kernel = cuda.cuModuleGetFunction(module, kernel_name)
    assert_drv(err)

    return kernel


@cuda_python_required
def assert_compile_error(err, prog):
    assert isinstance(err, nvrtc.nvrtcResult)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        error_log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, error_log)
        raise RuntimeError(f"Nvrtc Error: {err}\n" f"Compile log:\n{error_log.decode()}")


@cuda_python_required
def get_kernels(cuda_src: str, kernel_names: list[str]):
    if not torch.cuda.is_initialized():
        torch.cuda.init()
        torch.cuda.synchronize()

    # create PTX
    program_name = "-".join(kernel_names) + ".cu"
    err, prog = nvrtc.nvrtcCreateProgram(cuda_src.encode(), program_name.encode(), 0, [], [])

    # ? specify GPU arch
    # https://stackoverflow.com/questions/48283009/nvcc-get-device-compute-capability-in-runtime
    opts = []
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    assert_compile_error(err, prog)

    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    assert_drv(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    # load module
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    assert_drv(err)
    kernels = []
    for kernel_name in kernel_names:
        err, kernel = cuda.cuModuleGetFunction(module, kernel_name.encode())
        assert_drv(err)
        kernels.append(kernel)
    return kernels


@cuda_python_required
def get_kernel(cuda_src: str, kernel_name: str):
    (kernel,) = get_kernels(cuda_src=cuda_src, kernel_names=[kernel_name])
    return kernel


@cuda_python_required
def launch_kernel(
    kernel,
    gridDimX: int,
    gridDimY: int,
    gridDimZ: int,
    blockDimX: int,
    blockDimY: int,
    blockDimZ: int,
    kernel_args: list[np.ndarray],
    sharedMemBytes: int = 0,
    stream: int | None = None,
):
    # launch kernel
    args = np.array([arg.ctypes.data for arg in kernel_args], dtype=np.uint64)
    (err,) = cuda.cuLaunchKernel(
        kernel,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        stream if stream is not None else torch.cuda.default_stream().cuda_stream,
        args.ctypes.data,  # kernel args
        0,  # extra
    )
    assert_drv(err)
