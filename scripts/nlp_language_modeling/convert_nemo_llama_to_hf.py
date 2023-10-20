import re
import os
import tempfile
import torch

from collections import OrderedDict

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import (
    CustomProgressBar,
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.model_utils import inject_model_parallel_rank

@hydra_runner(config_path="conf", config_name="config_llama_truncate")
def main(cfg) -> None:
    plugins = []
    
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = None
    if cfg.trainer.precision == 16:
        trainer = Trainer(
            plugins=[
                NLPMixedPrecisionPlugin(
                    init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                    growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                ),
            ],
            strategy=NLPDDPStrategy(),
            **cfg.trainer,
        )
    elif cfg.trainer.precision == 'bf16':
        trainer = Trainer(plugins=plugins, strategy=NLPDDPStrategy(), **cfg.trainer,)
    else:
        trainer = Trainer(plugins=[NLPPrecisionPlugin()], strategy=NLPDDPStrategy(), **cfg.trainer)

    model = MegatronGPTModel.restore_from(
        cfg.restore_from_path, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector(),
    )

    param_to_weights = lambda param: param.float()
    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()
    
    hidden_size = cfg.model.hidden_size
    head_num = cfg.model.num_attention_heads
    num_layers = cfg.model.num_layers
    ffn_hidden_size = cfg.model.ffn_hidden_size
    head_size = hidden_size // head_num
    
    # Embedding
    embed_weight = model.state_dict()[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(embed_weight)
    
    for l in range(int(num_layers)):
        print(f"converting layer {l}")
        
        qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        q_weight = torch.empty(head_num * head_size, hidden_size)
        k_weight = torch.empty(head_num * head_size, hidden_size)
        v_weight = torch.empty(head_num * head_size, hidden_size)
        
        idx = 0
        while (head_size * idx < hidden_size):
            q_weight[head_size * idx : head_size * (idx+1), :] = qkv_weights[idx * (3 * head_size): idx * (3 * head_size) + head_size, :]
            k_weight[head_size * idx : head_size * (idx+1), :] = qkv_weights[idx * (3 * head_size) + head_size: idx * (3 * head_size) + (2 * head_size), :]
            v_weight[head_size * idx : head_size * (idx+1), :] = qkv_weights[idx * (3 * head_size) + (2 * head_size): idx * (3 * head_size) + (3 * head_size), :]
            idx += 1
            
        q_weights_base_name = f'model.layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'model.layers.{l}.self_attn.k_proj.weight' 
        v_weights_base_name = f'model.layers.{l}.self_attn.v_proj.weight'
        
        checkpoint['state_dict'][q_weights_base_name] = param_to_weights(q_weight)
        checkpoint['state_dict'][k_weights_base_name] = param_to_weights(k_weight)
        checkpoint['state_dict'][v_weights_base_name] = param_to_weights(v_weight)
        
        # attention dense
        o_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj.weight']
        o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
        checkpoint['state_dict'][o_weight_base_name] = param_to_weights(o_weight)
        
        # mlp
        mlp_weights = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
        mlp_down_proj_weight = torch.empty(ffn_hidden_size, hidden_size)
        mlp_gate_proj_weight = torch.empty(ffn_hidden_size, hidden_size)
        mlp_down_proj_weight = mlp_weights[ : ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size : , :]
        
        mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight' 
        
        checkpoint['state_dict'][mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint['state_dict'][mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)
        
        mlp_up_proj_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2.weight']
        mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        checkpoint['state_dict'][mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)
        
        #layernorm
        input_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        checkpoint['state_dict'][input_ln_base_name] = param_to_weights(input_ln_weight)
        
        post_attn_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        checkpoint['state_dict'][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)
        
        print(f"done layer {l}")
        
    final_ln_weight = model.state_dict()[f'model.decoder.final_layernorm.weight']
    final_ln_base_name = f'model.norm.weight'
    checkpoint['state_dict'][final_ln_base_name] = param_to_weights(final_ln_weight)
    
    output_layer_weight = model.state_dict()[f'model.output_layer.weight']
    output_layer_base_name = f'lm_head.weight'
    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)
    
    path = "/workspace/pretrain/nemo2huggingface-main/00_llama-hf-converted"
    torch.save(checkpoint, os.path.join(path, 'ver4_pytorch_model.bin'))


if __name__ == '__main__':
    main()