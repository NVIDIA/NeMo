from typing import Path
import inspect

from nemo.lightning.io.artifact import Artifact
from nemo.lightning.io.to_config import to_config

import fiddle as fdl


class HFAutoArtifact(Artifact):
    def dump(
        self, instance, value: Path, absolute_dir: Path, relative_dir: Path
    ) -> Path:
        instance.save_pretrained(Path(absolute_dir) / "artifacts")

        return "./" + str(Path(relative_dir) / "artifacts")

    def load(self, path: Path) -> Path:
        return path


def from_pretrained(auto_cls, pretrained_model_name_or_path="dummy"):
    return auto_cls.from_pretrained(pretrained_model_name_or_path)


@to_config.register(
    lambda v: not inspect.isclass(v)
    and getattr(v, "__module__", "").startswith("transformers")
    and hasattr(v, "save_pretrained")
    and hasattr(v, "from_pretrained")
)
def handle_hf_pretrained(value):
    return fdl.Config(
        from_pretrained,
        auto_cls=value.__class__,
        pretrained_model_name_or_path="dummy",
    )


from_pretrained.__io_artifacts__ = [HFAutoArtifact("pretrained_model_name_or_path")]
