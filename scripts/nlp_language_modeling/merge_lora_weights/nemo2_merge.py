from nemo import lightning as nl
from nemo.collections import llm
from typing import Any, Dict, List
import torch
from nemo.lightning.io import load_context, ModelConnector
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.utils.get_rank import is_global_rank_zero
from pathlib import Path
from nemo.utils import logging


def merge(base_model_state_dict: Dict[str, Any],
          lora_state_dict: Dict[str, Any],
          num_layers: int,
          tp_size: int,
          rank: int):
    mcore_layer_to_lora = {}


    """
    'self_attention.linear_qkv.adapter.linear_in.weight' 
    'self_attention.linear_qkv.adapter.linear_out.weight', 
    'self_attention.linear_proj.adapter.linear_in.weight'
    'self_attention.linear_proj.adapter.linear_out.weight',
    'mlp.linear_fc1.adapter.linear_in.weight',
    'mlp.linear_fc1.adapter.linear_out.weight', 
    'mlp.linear_fc2.adapter.linear_in.weight',
    'mlp.linear_fc2.adapter.linear_out.weight', 
    """

    mcore_layer_to_lora["attention_qkv"] = {
        "base_model_layer": "self_attention.linear_qkv.weight",
        "lora_in": "self_attention.linear_qkv.adapter.linear_in.weight",
        "lora_out": "self_attention.linear_qkv.adapter.linear_out.weight",
    }
    mcore_layer_to_lora["attention_dense"] = {
        "base_model_layer": "self_attention.linear_proj.weight",
        "lora_in": "self_attention.linear_proj.adapter.linear_in.weight",
        "lora_out": "self_attention.linear_proj.adapter.linear_out.weight",
    }
    mcore_layer_to_lora["mlp_fc1"] = {
        "base_model_layer": "mlp.linear_fc1.weight",
        "lora_in": "mlp.linear_fc1.adapter.linear_in.weight",
        "lora_out": "mlp.linear_fc1.adapter.linear_out.weight",
    }
    mcore_layer_to_lora["mlp_fc2"] = {
        "base_model_layer": "mlp.linear_fc2.weight",
        "lora_in": "mlp.linear_fc2.adapter.linear_in.weight",
        "lora_out": "mlp.linear_fc2.adapter.linear_out.weight",
    }

    for nl in range(num_layers):
        for key in mcore_layer_to_lora.keys():
            ##TODO: prefix should be model or module or 0.module?
            key_base = f'0.module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["base_model_layer"]}'
            key_lora_in = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_in"]}'
            key_lora_out = f'module.decoder.layers.{nl}.{mcore_layer_to_lora[key]["lora_out"]}'
            if key_lora_in in lora_state_dict and key_lora_out in lora_state_dict:
                if tp_size > 1:
                    gathered_lora_in = [torch.zeros_like(lora_state_dict[key_lora_in]) for _ in range(tp_size)]
                    gathered_lora_out = [torch.zeros_like(lora_state_dict[key_lora_out]) for _ in range(tp_size)]
                    torch.distributed.all_gather(gathered_lora_in, lora_state_dict[key_lora_in])
                    torch.distributed.all_gather(gathered_lora_out, lora_state_dict[key_lora_out])

                    if is_global_rank_zero():
                        print(f"RANK{torch.distributed.get_rank()} has {key_lora_in} shape {lora_state_dict[key_lora_in].shape}") #gathered lorain{gathered_lora_in}")
                        print(f"RANK{torch.distributed.get_rank()} has {key_lora_out} shape {lora_state_dict[key_lora_out].shape}") #gathered loraout {gathered_lora_out}")
                    ## TODO: Who decides what dim they split?
                    tp_dim_lora_in = 1 if key in ["attention_dense", 'mlp_fc2'] else 0
                    wt_lora_in = torch.cat(gathered_lora_in, dim=tp_dim_lora_in).float()
                    wt_lora_out = torch.cat(gathered_lora_out, dim=0).float()
                    wt_lora = wt_lora_out @ wt_lora_in
                    tp_dim_base = 0 if key in ["attention_qkv", "mlp_fc1"] else 1
                    wt_lora_current_rank = torch.chunk(wt_lora, tp_size, dim=tp_dim_base)[rank]
                else: #when tp==1
                    wt_lora_in = lora_state_dict[key_lora_in]
                    wt_lora_out = lora_state_dict[key_lora_out]
                    wt_lora = wt_lora_out @ wt_lora_in
                    wt_lora_current_rank = wt_lora

                wt_base = base_model_state_dict[key_base]
                logging.info(f"Full {key_base} wt_lora_in {wt_lora_in.shape}, wt_lora_out {wt_lora_out.shape}, wt_lora {wt_lora.shape}, wt_base {wt_base.shape}")

                
                base_model_state_dict[key_base] = (wt_base.float() + wt_lora_current_rank.to(wt_base.device)).type_as(wt_base)
                logging.info(f'merging for weight {key_base}')

    return base_model_state_dict


if __name__ == '__main__':
    tp_size = 1
    output_path= f"merged_ckpt_test_tp{tp_size}"
    adapter_path = Path("/workspace/peftmerge/exp/peft/nemo2_peft/checkpoints/nemo2_peft--reduced_train_loss=0.3022-epoch=0-last/")
    base_path = "/root/.cache/nemo/models/meta-llama/Meta-Llama-3-8B/"

    #==========Use connector.nemo_load directly NO=====
    # connector = ModelConnector()
    # connector.nemo_load(base_path, cpu=True) 
    #==================================================

    #==========Directly using dist_ckpt API YES============
    # from megatron.core import dist_checkpointing

    # trainer = nl.Trainer(
    #     devices=2,
    #     strategy=nl.MegatronStrategy(tensor_model_parallel_size=2),
    #     plugins=nl.MegatronMixedPrecision(precision='16-mixed'),
    #     accelerator="gpu",
    #     )
    # trainer.strategy.setup_environment()
    # model = load_context(base_path).model
    # model.configure_model()
    # mpmodel = MegatronParallel(model)
    
    # base_state_dict = dist_checkpointing.load(
    #     sharded_state_dict=mpmodel.sharded_state_dict(), checkpoint_dir=base_path
    # )
    # if is_global_rank_zero():
    #     print("===========")
    #     pks = ["module.decoder.layers.31.self_attention.linear_qkv.weight", 
    #            "module.decoder.layers.31.self_attention.linear_proj.weight",
    #            "module.decoder.layers.31.mlp.linear_fc1.weight",
    #            "module.decoder.layers.31.mlp.linear_fc2.weight"]
    #     for pk in pks:
    #         print(pk, base_state_dict[pk].shape)
    #     exit()    
    #====================================================


    #==========Using fabric API==========================
    trainer = nl.Trainer(
        devices=tp_size,
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=tp_size),
        plugins=nl.MegatronMixedPrecision(precision='16-mixed'),
        accelerator="cpu",
        )
    fabric = trainer.to_fabric()
    #load base model state dict
    model = fabric.load_model(base_path)
    base_state_dict=model.state_dict()
    print("Loaded base model state dict")
    #======================================================

    #To get lora sharded state dict
    context = load_context(adapter_path / 'context')
    if not context.model.state_dict():
        context.model.configure_model()
    dummymodel = MegatronParallel(context.model)
    lora = llm.peft.LoRA() ## TODO: Where is LoRA related configs stored? Not in context
    dummymodel.freeze()
    dummymodel.walk(lora.transform)
    def adapter_key_filter(key: str) -> bool:
        return ".adapter." in key or key.endswith(".adapters")
    adapter_sharded_state_dict={}
    adapter_sharded_state_dict['state_dict'] = {
        k: v for k, v in dummymodel.sharded_state_dict().items() if adapter_key_filter(k)
    }
    map_location=None
    adapter_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(adapter_path / "weights", adapter_sharded_state_dict, map_location)
    lora_state_dict = {k: v for k,v in adapter_ckpt['state_dict'].items() if v is not None} #filter extra_state which has None value
    print("Loaded lora state dict")
    
    
    merged_weights = merge(base_state_dict, lora_state_dict, 32, tp_size=tp_size, rank=torch.distributed.get_rank())
    model.load_state_dict(merged_weights)

    trainer.strategy.checkpoint_io.save_checkpoint(model.sharded_state_dict(), output_path)
    if is_global_rank_zero():
        model.io_dump(output_path)
    



