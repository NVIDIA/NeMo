import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS



from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule

# logging.setLevel(logging.DEBUG)
from transformers import MllamaForConditionalGeneration, AutoProcessor
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm.llama.data.task_encoder import LlamaTaskEncoder
from transformers import AutoTokenizer
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios

from nemo import lightning as nl
from nemo.collections import vlm

def get_aspect_ratio(aspect_ratio_ids):
    max_image_tiles = 4
    mapping = get_all_supported_aspect_ratios(max_image_tiles)
    aspect_ratio = [mapping[i.item()-1] for i in aspect_ratio_ids]
    return torch.tensor(aspect_ratio)

def get_train_dataloader() -> TRAIN_DATALOADERS:
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

    task_encoder = LlamaTaskEncoder(
        tokenizer=tokenizer, image_processor=image_processor, multimodal_sample_config=multimodal_sample_config
    )
    data_module = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_workers=0,
        micro_batch_size=16,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    train_loader = data_module.train_dataloader()
    return train_loader


def get_model() -> vlm.MLlamaModel:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        ckpt_load_optimizer=False,
        ckpt_save_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=2,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
    model = vlm.MLlamaModel(
        vlm.MLlamaModelConfig(
            language_model_config=vlm.CrossAttentionTextModelConfig8B(rotary_interleaved=True, apply_rope_fusion=False),
            vision_model_config=vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280,
                                                                    num_attention_heads=16, vision_chunk_size=448,
                                                                    vision_max_num_chunks=4, ),
            # vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,),
        ),
        tokenizer=tokenizer)

    fabric = trainer.to_fabric()
    # local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/evian3-11b-vision-final_vv1_zarr/"
    local_model_path = "/lustre/fsw/coreai_dlalgo_llm/nemo_home/models/meta-llama/Llama-3.2-11B-Vision_zarr"
    model = fabric.load_model(local_model_path, model)

    model = model.module.cuda()
    model.eval()
    model = model.to(torch.bfloat16)
    return model


if __name__ == '__main__':

    train_loader = get_train_dataloader()
    model: vlm.MLlamaModel = get_model()

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            output = model(
                batch_images=batch["media"].cuda(non_blocking=True),
                batch_masks=batch["vision_mask"].cuda(non_blocking=True),
                aspect_ratios=get_aspect_ratio(batch["aspect_ratio_ids"]).cuda(non_blocking=True),
                tokens=batch["tokens"].cuda(non_blocking=True),
                position_ids=batch["position_ids"].cuda(non_blocking=True),
            )

        # print("keys  ", batch["__keys__"])
        print("aspect_ratio_ids", batch["aspect_ratio_ids"])
        # print("image tensor shape", batch["media"].shape)
        # print("prompt tokens tensor shape", batch["tokens"].shape)
        # print("labels tensor shape", batch["labels"].shape)
        # print("loss mask shape", batch["loss_mask"].shape)
        # print(batch.keys())
        # print("************************************")
        # print(f"labels 0 shape", batch["labels"][0])
        # print("**********************************")
        # print(f"loss_mask   {batch['loss_mask']}")
        # print("**********************************")
        # print(f"vision mask {batch['vision_mask']}")
        # print("**********************************")
        if i == 0:
            break

