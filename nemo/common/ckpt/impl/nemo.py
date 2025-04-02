import fiddle as fdl
from nemo_run import Config, Partial
from pathlib import Path
from fsspec.spec import AbstractFileSystem
import torch.nn as nn

from nemo.common.ckpt.registry import CheckpointHandler, registry, load_context, detect_checkpoint_type
from nemo_run.run.experiment import maybe_load_external_main
from nemo.common.ckpt.impl.megatron import InitMegatronModel


try:
    from nemo.lightning.io.api import load

    @registry.register("nemo")
    class NemoHandler(CheckpointHandler):
        def detect(self, fs: AbstractFileSystem, path: str, files: list[str]) -> bool:
            if "context/io.json" not in files:
                return False
            
            return any(f.startswith("weights/") for f in files)

        def load_context(self, path: str) -> Config:
            maybe_load_external_main(Path(path) / "context")
            out = load(Path(path) / "context", build=False)

            if isinstance(out, fdl.Config):
                return fdl.cast(Config, out)
            elif isinstance(out, fdl.Partial):
                return fdl.cast(Partial, out)

            raise ValueError("Expected a Fiddle Config object")
        
        def load_model(self, path: str) -> nn.Module:
            # TODO: Detect mcore dist-checkpoint and only then use InitMegatronModel
            return InitMegatronModel(path, setup=self.load_context(path))

except ImportError:
    pass


if __name__ == "__main__":
    path = "/workspaces/nv/work/auto_model/nemo_experiments/automodel_trainer/2025-03-10_16-01-52/checkpoints/"
    
    print(detect_checkpoint_type(path, path_resolver="latest"))
    cfg = load_context(path, path_resolver="latest")
    print(cfg)