from pathlib import Path

from nemo.collections import llm, vlm
from nemo.collections.llm import import_ckpt

if __name__ == '__main__':
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = vlm.Qwen2VLModel(vlm.Qwen2VLConfig2B())
    nemo2_path = import_ckpt(model=model, source=f'hf://{model_id}', output_path=Path('./qwen2vl2B_nemo_ckpt'))
