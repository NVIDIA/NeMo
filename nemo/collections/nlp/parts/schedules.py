# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union, List, Optional, Sequence

import torch

from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.pipeline_parallel.utils import get_kth_microbatch
from apex.transformer.pipeline_parallel.utils import listify_model
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from apex.transformer.pipeline_parallel.utils import get_model_type
from apex.transformer.pipeline_parallel.schedules.common import Batch
from apex.transformer.pipeline_parallel.schedules.common import FwdStepFunc
from apex.transformer.pipeline_parallel.schedules.common import forward_step
from apex.transformer.pipeline_parallel.schedules.common import backward_step
from apex.transformer.log_util import get_transformer_logger


_logger = get_transformer_logger(__name__)


def get_tensor_shapes(
    rank: int,
    model_type: ModelType,
    *,
    tensor_shape: Union[List[int], torch.Size],
    decoder_sequence_length: Optional[int] = None,
) -> Sequence[Sequence[int]]:
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"
    sequence_length, micro_batch_size, hidden_size = tensor_shape
    tensor_shapes = []
    if model_type == ModelType.encoder_and_decoder:
        if decoder_sequence_length is None:
            raise ValueError("`decoder_sequence_length` is required for `ModelType.encoder_and_decoder`")
        if parallel_state.is_pipeline_stage_before_split(rank):
            # If next rank is after split, then need transpose for encoder_hidden_state.
            if parallel_state.is_pipeline_stage_before_split(rank + 1):
                tensor_shapes.append((sequence_length, micro_batch_size, hidden_size))
            else:
                tensor_shapes.append((micro_batch_size, sequence_length, hidden_size))
        else:
            tensor_shapes.append((decoder_sequence_length, micro_batch_size, hidden_size))
            tensor_shapes.append((micro_batch_size, sequence_length, hidden_size))
    else:
        tensor_shapes.append((sequence_length, micro_batch_size, hidden_size))
    return tensor_shapes


def recv_forward(
    tensor_shapes: List[Union[None, List[int]]], *, dtype: Optional[torch.dtype] = None,
) -> List[Union[None, torch.Tensor]]:
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape=tensor_shape, dtype=dtype))
    return input_tensors


def recv_backward(
    tensor_shapes: List[Union[None, List[int]]], *, dtype: Optional[torch.dtype] = None,
) -> List[Union[None, torch.Tensor]]:
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape=tensor_shape, dtype=dtype))
    return output_tensor_grads


def send_forward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)


def send_backward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
) -> None:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)


def send_forward_recv_backward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
) -> List[Union[None, torch.Tensor]]:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape=tensor_shape, dtype=dtype
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
    *,
    dtype: Optional[torch.dtype] = None,
) -> List[Union[None, torch.Tensor]]:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype
        )
        input_tensors.append(input_tensor)
    return input_tensors


def nemo_forward_backward_pipelining_without_interleaving(
    forward_step_func: FwdStepFunc,
    batch: Batch,
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    *,
    forward_only: bool,
    tensor_shape: Optional[Union[List[int], torch.Size]] = None,
    decoder_sequence_length: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    disable_autocast: bool = False,
    return_output_tensor: bool = False,
) -> List[Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Run non-interleaved 1F1B schedule, with communication between pipeline stages.

    This pipeline parallel scheduling consists of three steps:
        1. warmup
        2. 1F1B a.k.a. steady state
        3. cooldown if not forward_only

    Args:
        forward_step_func: A function which takes a minibatch and model as its arguments and
            returns model's forward output and the loss function.
            The loss function is supposed to take one `torch.Tensor` and
            return a `torch.Tensor` of loss and a dictionary of `str` and `torch.Tensor`.
        batch: A minibatch, i.e., a list of `torch.Tensor`'s.
        model: A `torch.nn.Module` or a list of `torch.nn.Module`.

    Keyword args:
        forward_only:
        tensor_shape: Shape of tensor. Required for P2P communication.
        dtype: dtype used in p2p communication. If ``None`` (default value),
            torch.float32 will be used even if ``autocast`` is enabled.

    Returns:
        a list of loss `torch.Tensor`s if the last stage, empty list otherwise.
    """
    # timers = get_timers()

    model: List[torch.nn.Module] = listify_model(model)
    if len(model) != 1:
        msg = f"`model` is expected be a `nn.Module`, but {type(model)}"
        raise RuntimeError(msg)
    model: torch.nn.Module = model[0]

    # Compute number of warmup microbatches.
    num_microbatches: int = get_num_microbatches()
    num_warmup_microbatches: int = (
        parallel_state.get_pipeline_model_parallel_world_size() - parallel_state.get_pipeline_model_parallel_rank() - 1
    )
    num_warmup_microbatches: int = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining: int = num_microbatches - num_warmup_microbatches

    model_type = get_model_type(model)
    rank: int = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes: List[List[int]] = get_tensor_shapes(
        rank - 1, model_type, tensor_shape=tensor_shape, decoder_sequence_length=decoder_sequence_length
    )
    send_tensor_shapes: List[List[int]] = get_tensor_shapes(
        rank, model_type, tensor_shape=tensor_shape, decoder_sequence_length=decoder_sequence_length
    )

    _logger.info(
        f"num_microbatches: {num_microbatches}, "
        f"num_warmup_microbatches: {num_warmup_microbatches}, "
        f"num_microbatches_remaining: {num_microbatches_remaining}"
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors: List[Union[None, torch.Tensor]] = []
    output_tensors: List[Union[None, torch.Tensor]] = []
    losses_reduced: List[Union[None, torch.Tensor]] = []
    ###################################################################################################################
    # Run warmup forward passes.
    ###################################################################################################################
    _logger.info("Warmup")
    for i in range(num_warmup_microbatches):
        _logger.debug(f"warmup iter: {i} / {num_warmup_microbatches}")
        _logger.debug("receive fwd")
        input_tensor = recv_forward(tensor_shapes=recv_tensor_shapes, dtype=dtype)
        cur_microbatch = get_kth_microbatch(batch, i)
        output_tensor = forward_step(
            forward_step_func, cur_microbatch, model, input_tensor, losses_reduced, dtype, disable_autocast,
        )
        _logger.debug("send fwd")
        send_forward(output_tensor, tensor_shapes=send_tensor_shapes, dtype=dtype)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        _logger.debug("recv_forward before steady state start")
        input_tensor: List[Union[None, torch.Tensor]] = recv_forward(tensor_shapes=recv_tensor_shapes, dtype=dtype)

    ###################################################################################################################
    # Run 1F1B in steady state.
    ###################################################################################################################
    _logger.info("Steady phase")
    for i in range(num_microbatches_remaining):
        _logger.debug(f"steady iter: {i} / {num_microbatches_remaining}")
        last_iteration: bool = i == (num_microbatches_remaining - 1)

        cur_microbatch: torch.Tensor = get_kth_microbatch(batch, i + num_warmup_microbatches)
        output_tensor: Union[torch.Tensor, Sequence[torch.Tensor]] = forward_step(
            forward_step_func, cur_microbatch, model, input_tensor, losses_reduced, dtype, disable_autocast,
        )
        if forward_only:
            _logger.debug("send fwd")
            send_forward(output_tensor, tensor_shapes=send_tensor_shapes, dtype=dtype)

            if not last_iteration:
                _logger.debug("receive fwd (last iteration)")
                input_tensor = recv_forward(tensor_shapes=recv_tensor_shapes, dtype=dtype)

        else:
            _logger.debug("send fwd & receive bwd")
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, tensor_shapes=send_tensor_shapes, dtype=dtype
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type=model_type, grad_scaler=grad_scaler
            )

            if last_iteration:
                input_tensor = None
                _logger.debug("send bwd")
                send_backward(input_tensor_grad, tensor_shapes=recv_tensor_shapes, dtype=dtype)
            else:
                _logger.debug("send bwd and receive fwd")
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, tensor_shapes=recv_tensor_shapes, dtype=dtype
                )
    ###################################################################################################################
    # Run cooldown backward passes.
    ###################################################################################################################
    _logger.info("Cooldown phase")
    if not forward_only:
        for i in range(num_warmup_microbatches):
            _logger.debug(f"cooldown iter: {i} / {num_warmup_microbatches}")
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            _logger.debug("receive bwd")
            output_tensor_grad = recv_backward(tensor_shapes=send_tensor_shapes, dtype=dtype)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type=model_type, grad_scaler=grad_scaler
            )

            _logger.debug("send bwd")
            send_backward(input_tensor_grad, tensor_shapes=recv_tensor_shapes, dtype=dtype)

    if return_output_tensor:
        return output_tensors

    return losses_reduced
