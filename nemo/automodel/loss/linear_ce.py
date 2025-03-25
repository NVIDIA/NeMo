
import torch
from nemo.utils.import_utils import safe_import_from


linear_cross_entropy, HAVE_LINEAR_LOSS_CE = safe_import_from(
    "cut_cross_entropy",
    "linear_cross_entropy",
)



def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    lm_weight: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    logit_softcapping: float = 0,
    accuracy_threshold: str = "auto",
):
    """
    Compute fused linear cross entropy loss that matches PyTorch's cross_entropy behavior.

    Args:
        hidden_states: Input hidden states
        lm_weight: Weight matrix for linear transformation
        labels: Target labels
        num_items_in_batch: Number of valid tokens (where labels != ignore_index)
        ignore_index: Value to ignore in labels (default: -100)
        reduction: Reduction method ('mean' or 'sum')
        logit_softcapping: Value for softcapping logits (0 means no capping)
        accuracy_threshold: Threshold for accuracy computation
    """
    # First compute loss with sum reduction to handle normalization ourselves
    if logit_softcapping == 0:
        logit_softcapping = None

    # Compute loss with shift=False to match PyTorch behavior
    # Set filter_eps=None to avoid any token filtering
    loss = linear_cross_entropy(
        hidden_states,
        lm_weight,
        targets=labels,
        ignore_index=ignore_index,
        softcap=logit_softcapping,
        reduction="sum",  # Use sum reduction to handle normalization ourselves
        shift=False,  # Match PyTorch behavior
        filter_eps=None,  # No token filtering
    )

    # Match PyTorch's cross_entropy behavior:
    # For mean reduction, divide by number of valid tokens
    if reduction == "mean":
        if num_items_in_batch is None:
            num_items_in_batch = torch.sum(labels != ignore_index).item()
        loss = loss / num_items_in_batch
    print(f"####### Linear CE loss: {loss}")
    return loss
