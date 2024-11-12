from pathlib import Path

from nemo.collections.llm import import_ckpt
from nemo.collections.llm.gpt.model.llama import Llama2Config7B, LlamaModel

if __name__ == "__main__":
    import_ckpt(
        model=LlamaModel(Llama2Config7B()),
        source='hf://llava-hf/llava-v1.6-vicuna-7b-hf',
        # output_path=Path('/workspace/'),
    )
