# /workspace/finetune_qwen2vl_recipe.py
from nemo.collections import vlm
import nemo_run as run
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from transformers import Qwen2VLImageProcessor
from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
from nemo.collections.vlm.qwen2vl.data.task_encoder import Qwen2VLTaskEncoder

finetune = vlm.recipes.qwen2vl_2b.finetune_recipe(
    name="qwen2vl_2b_finetune",
    dir="~/.cache/nemo/models/Qwen/Qwen2-VL-2B-Instruct",
    num_nodes=1,
    num_gpus_per_node=1,  # 사용 GPU 개수
    peft_scheme="None",   # 풀파인튜닝 원하면 'none'
)

# 2) 데이터 모듈: Energon(WDS + .nv-meta) 경로 지정
DATA_ROOT = "/datasets/wds"  # 'energon prepare'까지 끝난 폴더

tokenizer = AutoTokenizer("Qwen/Qwen2-VL-2B-Instruct")
image_processor = Qwen2VLImageProcessor()

finetune.data = EnergonMultiModalDataModule(
    path=DATA_ROOT,
    tokenizer=tokenizer,
    image_processor=image_processor,
    seq_length=4096,
    micro_batch_size=1,
    global_batch_size=2,
    num_workers=8,
    task_encoder=Qwen2VLTaskEncoder(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_padding_length=int(4096 * 0.9),
    ),
)

# 3) 실행
if __name__ == "__main__":
    run.run(finetune)  # 보통 로컬 멀티GPU 자동 실행
