import argparse
import os
import sys
import logging
import torch
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import import_ckpt
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.mimo.data.captioning import MimoCaptioningTaskEncoder
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig, BaseMimoModel, CustomMimoConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

def main(args):
    # Global and micro batch sizes
    gbs = 1
    mbs = 1
    seq_length = 256
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )
    data_path = '/lustre/fsw/coreai_dlalgo_genai/ykarnati/datasets/cc3m-wds'
    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    multimodal_sample_config = MultiModalSampleConfig()

    task_encoder = MimoCaptioningTaskEncoder(
        tokenizer=tokenizer.tokenizer,
        image_processor=processor.image_processor,
        multimodal_sample_config=multimodal_sample_config,
    )
    data = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        num_workers=8,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    train_loader = data.train_dataloader()
    one_batch = next(iter(train_loader))
    
    fabric = trainer.to_fabric()

    custom_config = CustomMimoConfig(
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
    )
    # base_config = BaseMimoConfig(vocab_size = tokenizer.vocab_size)
    model = BaseMimoModel(config=custom_config, tokenizer=tokenizer)
    model = fabric.load_model(args.local_model_path, model)
    
    model = model.module.cuda()
    model.eval()


    images = one_batch["images"].cuda()
    input_ids = one_batch["tokens"].cuda()
    position_ids = one_batch["position_ids"].cuda()
    input_text = one_batch['input_text']
    
    
    all_hidden_states = []
    
   
    input_ids = input_ids[:,:-7]
    position_ids = position_ids[:,:-7]
    # if torch.distributed.get_rank() == 0: #or other ranks
    #     breakpoint()
    # torch.distributed.barrier()
    generated_ids = input_ids.clone()
    for _ in range(8):
        with torch.no_grad():

            output_dict = model(
                input_ids = input_ids,
                images=images,
                input_text=input_text,
                position_ids=position_ids,
                attention_mask=None,
                # labels = labels,
                # loss_mask = loss_mask
                # num_media_tiles=num_media_tiles,
            )
            output = output_dict['output']
            # output_projection_embeddings =  output_dict['output_projection_embeddings']
            # image_caption_embeddings =  output_dict['image_caption_embeddings']
            hiden_states = output_dict['hidden_states']

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            # if torch.distributed.get_rank() == 0: #or other ranks
            #     breakpoint()
            # torch.distributed.barrier()
            all_hidden_states.append(hiden_states[-1,:,:])
           
            # If the generated token is the end of sequence token, stop generating
            # if next_token_ids.item() == tokenizer.eos_token_id:
            #     break
    # if torch.distributed.get_rank() == 0: #or other ranks
    #     breakpoint()
    # torch.distributed.barrier()
    hidden_states_concat = torch.cat(all_hidden_states, dim = 0).unsqueeze(0)
    vis_proj_out = model.module.module.module.vision_output_projection_module(hidden_states_concat)
    actual_image_caption_embeddings =  model.module.module.module.get_image_caption_embeddings(one_batch['input_text'])
    mse_loss = torch.nn.functional.mse_loss(actual_image_caption_embeddings.to(vis_proj_out.device, dtype = vis_proj_out.dtype), vis_proj_out)
    
   
    device = vis_proj_out.device
    image_decode_device = model.module.module.module.image_decoder.to(device)
    gen_image =image_decode_device(prompt_embeds=actual_image_caption_embeddings.to(device)).images[0]
    gen_image.save('debug_image_gt.png')
    
    gen_image = image_decode_device(prompt_embeds=vis_proj_out).images[0]
    gen_image.save('debug_image_generated.png')
    
    logging.info(f"MSE loss for embeddings {mse_loss}")
    generated_ids[generated_ids == -200] = 0
    generated_texts = tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    logging.info("======== GENERATED TEXT OUTPUT ========")
    logging.info(f"{generated_texts}")
    logging.info("=======================================")
   

    # Optimizer and scheduler setup
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimo generation example Script")

    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )

    args = parser.parse_args()
    main(args)