import fabric
import torch

from nemo.utils import logging


from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule

# logging.setLevel(logging.DEBUG)
from transformers import MllamaForConditionalGeneration, AutoProcessor
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder
from transformers import AutoTokenizer

from nemo import lightning as nl
from nemo.collections import vlm



def get_dataloader():
    data_path = '/lustre/fsw/coreai_dlalgo_genai/datasets/energon_datasets/LLaVA-Pretrain-LCS-558K'
    model_directory = "/lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-instruct-final-hf_vv1/"
    # model_id = "evian3-11b-vision-instruct-final-hf_vv1"
    processor = AutoProcessor.from_pretrained(model_directory)
    image_processor = processor.image_processor
    image_processor.size = {'height': 448, 'width': 448}

    tokenizer = processor.tokenizer

    multimodal_sample_config = MultiModalSampleConfig()
    multimodal_sample_config.conversation_template_config.system = "You are a helpful assistant"
    multimodal_sample_config.conversation_template_config.chat_template = None
    multimodal_sample_config.image_token.token_id = 128256
    multimodal_sample_config.conversation_template_config.stop_string = None


if __name__ == '__main__':

    train_loader = get_dataloader()
    for i, batch in enumerate(train_loader):
        # print("keys  ", batch["__keys__"])
        print("image tensor shape", batch["media"].shape)
        print("prompt tokens tensor shape", batch["tokens"].shape)
        print("labels tensor shape", batch["labels"].shape)
        print("loss mask shape", batch["loss_mask"].shape)
        print(batch.keys())
        # print("************************************")
        # print(f"labels 0 shape", batch["labels"][0])
        # print("**********************************")
        # print(f"loss_mask   {batch['loss_mask']}")
        # print("**********************************")
        # print(f"vision mask {batch['vision_mask']}")
        # print("**********************************")
        if i == 0:
            break

