from pathlib import Path

from nemo.collections.llm import import_ckpt
from nemo.collections.llm.gpt.model.starcoder2 import Starcoder2Config3B, Starcoder2Model

if __name__ == "__main__":
    import_ckpt(
        model=Starcoder2Model(Starcoder2Config3B()),
        source='hf://bigcode/starcoder2-3b',
        output_path=Path('/workspace/starcoder2_3b_nemo2'),
        overwrite=True,
    )
