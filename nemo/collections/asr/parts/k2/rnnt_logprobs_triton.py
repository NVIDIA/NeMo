import torch
import triton
import triton.language as tl


@triton.jit
def _rnnt_logprobs_fwd_kernel(
    x_ptr,
    targets_ptr,
    source_len: int,
    target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    target_scores_ptr,
    blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    flat_index = ((batch_i * source_len + source_i) * target_len_plus_1 + target_i) * num_labels
    x_ptr += flat_index
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(x_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    denominator = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    blank_logit = tl.load(x_ptr + blank_id).to(tl.float32)
    flat_index_output = (batch_i * source_len + source_i) * target_len_plus_1 + target_i
    tl.store(blank_scores_ptr + flat_index_output, blank_logit - logits_max - denominator)
    if target_i < target_len_plus_1 - 1:
        target_id = tl.load(targets_ptr + batch_i * (target_len_plus_1 - 1) + target_i)
        target_logit = tl.load(x_ptr + target_id).to(tl.float32)
        tl.store(target_scores_ptr + flat_index_output, target_logit - logits_max - denominator)


@triton.jit
def _rnnt_logprobs_bwd_kernel(
    x_ptr,
    grad_x_ptr,
    targets_ptr,
    source_len: int,
    target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    flat_index = ((batch_i * source_len + source_i) * target_len_plus_1 + target_i) * num_labels
    x_ptr += flat_index
    grad_x_ptr += flat_index

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(x_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    denominator = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    log_softmax = logits_minus_max - denominator
    softmax = tl.exp(log_softmax)

    flat_index_grad = (batch_i * source_len + source_i) * target_len_plus_1 + target_i
    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grad).to(tl.float32)
    target_i_valid = target_i < target_len_plus_1 - 1
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grad, mask=target_i_valid, other=0.0).to(tl.float32)
    target_id = tl.load(targets_ptr + batch_i * (target_len_plus_1 - 1) + target_i, mask=target_i_valid, other=-1)

    grad_not_in_targets = (-softmax) * (blank_grad + target_grad)
    grad = tl.where(col_offsets == blank_id, blank_grad + grad_not_in_targets, grad_not_in_targets)
    grad = tl.where(col_offsets == target_id, target_grad + grad_not_in_targets, grad)
    tl.store(grad_x_ptr + col_offsets, grad, mask=mask)


class RnntLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, targets: torch.Tensor, blank_id: int):
        assert x.is_contiguous()
        targets = targets.contiguous()
        device = x.device
        float_dtype = torch.float32

        target_scores = torch.empty(x.shape[:-1], dtype=float_dtype, device=device)
        blank_scores = torch.empty_like(target_scores)
        _rnnt_logprobs_fwd_kernel[(x.shape[0], x.shape[1], x.shape[2])](
            x_ptr=x,
            targets_ptr=targets,
            source_len=x.shape[1],
            target_len_plus_1=x.shape[2],
            num_labels=x.shape[3],
            blank_id=blank_id,
            target_scores_ptr=target_scores,
            blank_scores_ptr=blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(x.shape[-1]),
        )

        # saving for backward
        ctx.save_for_backward(x, targets)
        ctx.blank_id = blank_id
        return target_scores, blank_scores

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        # raise NotImplementedError
        (x, targets) = ctx.saved_tensors
        blank_id = ctx.blank_id
        grad_x = torch.zeros_like(x)
        _rnnt_logprobs_bwd_kernel[(x.shape[0], x.shape[1], x.shape[2])](
            x_ptr=x,
            grad_x_ptr=grad_x,
            targets_ptr=targets,
            source_len=x.shape[1],
            target_len_plus_1=x.shape[2],
            num_labels=x.shape[3],
            blank_id=blank_id,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(x.shape[-1]),
        )
        return grad_x, None, None


def rnnt_logprobs_triton(x: torch.Tensor, targets: torch.Tensor, blank_id: int):
    return RnntLogProbs.apply(x, targets, blank_id)
