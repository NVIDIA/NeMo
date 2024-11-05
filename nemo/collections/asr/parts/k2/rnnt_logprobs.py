import torch
import torch.nn.functional as F


def rnnt_logprobs_torch(x: torch.Tensor, targets: torch.Tensor, blank_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    batch_size = x.shape[0]
    x_log_softmax = F.log_softmax(x, dim=-1)
    blank_scores = x_log_softmax[..., blank_id]
    targets = torch.cat((targets, torch.zeros(batch_size, dtype=targets.dtype, device=device).unsqueeze(1)), dim=-1)
    target_scores = torch.gather(
        x_log_softmax, dim=-1, index=targets.unsqueeze(1).expand(x.shape[:-1]).unsqueeze(-1)
    ).squeeze(-1)
    return target_scores, blank_scores
