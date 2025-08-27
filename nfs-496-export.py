from pathlib import Path
from nemo.collections import llm
import os

# nemo_checkpoint_path = os.environ['CKPT_NEMO']  # Fail fast if env var undefined.
# nemo_checkpoint_path = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/checkpoints/qwen-3-32b-cpt/2025-08-19_01-27-25/checkpoints/model_name=0--val_loss=0.00-step=3-consumed_samples=16.0-last"
# nemo_checkpoint_path = "/lustre/fs1/portfolios/coreai/users/zhiyul/nemo/Qwen/Qwen3-32B"
# nemo_checkpoint_path = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/checkpoints/qwen-3-32b-cpt/2025-08-19_11-41-13/checkpoints/model_name=0--val_loss=0.00-step=3-consumed_samples=16.0-last"
nemo_checkpoint_path = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/checkpoints/qwen-3-32b-cpt/2025-08-19_12-20-36/checkpoints/model_name=0--val_loss=0.00-step=3-consumed_samples=16.0-last"
# nemo_checkpoint_path = "/lustre/fsw/portfolios/coreai/users/zhiyul/NeMo/checkpoints/qwen-3-32b-cpt/2025-08-19_13-56-05/checkpoints/model_name=0--val_loss=0.00-step=3-consumed_samples=16.0-last"
hf_checkpoint_path = "/tmp/my-qwen-3-32b-it"

if __name__ == "__main__":
    output_path = Path(hf_checkpoint_path)
    print(f"\nStarting export from {nemo_checkpoint_path} to {output_path}")
    result = llm.export_ckpt(
        path=Path(nemo_checkpoint_path),
        target="hf",
        output_path=output_path,
        overwrite=True,
    )
    print(f"\nExport completed with result: {result}")
