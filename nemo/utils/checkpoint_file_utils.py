import re

def parse_prefix_with_step(path: str) -> str:
    """
    Use regex to find the pattern up to "-step=900-"
    s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-validation_loss=6.47-consumed_samples=35960.0-last.ckpt
    should return s3://path/to/checkpoints/tp_rank_00_pp_rank_000/megatron_gpt--step=900-
    """
    match = re.search(r'(.*step=\d+-)', path)

    if match:
        return match.group(1)

    return path
