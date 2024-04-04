import os
from argparse import ArgumentParser

import torch
from omegaconf.omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.utils import logging
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


'''
CUDA_VISIBLE_DEVICES="0" python /home/ataghibakhsh/NeMo/scripts/checkpoint_converters/convert_griffin_to_mcore.py --output_path /home/ataghibakhsh/griffin_it.nemo --hparams_file /home/ataghibakhsh/NeMo/examples/nlp/language_modeling/conf/megatron_griffin_config.yaml --path_to_base /home/ataghibakhsh/deepmind/space_gemma_model/2b-it.pt
'''

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_griffin_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--path_to_base", type=str, default=None, required=True, help="Path to input .pt file.")
    args = parser.parse_args()
    return args


def convert(args):
    # logging.info(f"Loading checkpoint from HF: `{args.input_name_or_path}`")
    trainer = Trainer(
    strategy=NLPDDPStrategy(),
    devices=-1,
    accelerator="gpu",
    num_nodes=1,
    precision="bf16",
    logger=False,
    enable_checkpointing=False,
    use_distributed_sampler=False,
)
    # exp_manager(trainer, cfg.exp_manager)
    cfg = OmegaConf.load(args.hparams_file)
    model = MegatronGriffinModel(cfg.model, trainer)

    new_state_dict = {}
    dm_model_weight = torch.load(args.path_to_base)

    new_state_dict['model.embedder.word_embeddings.weight'] = dm_model_weight['embedder.input_embedding']
    new_state_dict['model.final_norm.weight'] = dm_model_weight['final_norm.scale']

    for l in range(26):
        print(f"Converting Layer {l}")
        print("********************")
        new_state_dict[f'model.layers.{l}.input_layernorm.weight'] = dm_model_weight[f'blocks.{l}.temporal_pre_norm.scale']

        new_state_dict[f'model.layers.{l}.mlp.linear_fc1.weight'] = torch.cat([dm_model_weight[f'blocks.{l}.mlp_block.ffw_up.w'].permute(0,2,1)[0], dm_model_weight[f'blocks.{l}.mlp_block.ffw_up.w'].permute(0,2,1)[1]])
        new_state_dict[f'model.layers.{l}.mlp.linear_fc1.bias'] = dm_model_weight[f'blocks.{l}.mlp_block.ffw_up.b'].flatten()
        new_state_dict[f'model.layers.{l}.mlp.linear_fc2.weight'] = dm_model_weight[f'blocks.{l}.mlp_block.ffw_down.weight']
        new_state_dict[f'model.layers.{l}.mlp.linear_fc2.bias'] = dm_model_weight[f'blocks.{l}.mlp_block.ffw_down.bias']
        new_state_dict[f'model.layers.{l}.mlp.linear_fc1._extra_state'] = model.state_dict()[f'model.layers.{l}.mlp.linear_fc1._extra_state']
        new_state_dict[f'model.layers.{l}.mlp.linear_fc2._extra_state'] = model.state_dict()[f'model.layers.{l}.mlp.linear_fc2._extra_state']

        new_state_dict[f'model.layers.{l}.pre_mlp_layernorm.weight'] = dm_model_weight[f'blocks.{l}.channel_pre_norm.scale']

        if l % 3 == 2:
            new_state_dict[f'model.layers.{l}.self_attention.linear_proj.weight'] = dm_model_weight[f'blocks.{l}.attention_block.proj_final.weight']
            new_state_dict[f'model.layers.{l}.self_attention.linear_proj.bias'] = dm_model_weight[f'blocks.{l}.attention_block.proj_final.bias']
            new_state_dict[f'model.layers.{l}.self_attention.linear_qkv.weight'] = torch.cat([dm_model_weight[f'blocks.{l}.attention_block.proj_q.weight'], 
                                                            dm_model_weight[f'blocks.{l}.attention_block.proj_k.weight'],
                                                            dm_model_weight[f'blocks.{l}.attention_block.proj_v.weight']])
            new_state_dict[f'model.layers.{l}.self_attention.linear_qkv.bias'] = torch.zeros(new_state_dict[f'model.layers.{l}.self_attention.linear_qkv.weight'].shape[0])
            new_state_dict[f'model.layers.{l}.self_attention.linear_proj._extra_state'] = model.state_dict()[f'model.layers.{l}.self_attention.linear_proj._extra_state']
            new_state_dict[f'model.layers.{l}.self_attention.linear_qkv._extra_state'] = model.state_dict()[f'model.layers.{l}.self_attention.linear_qkv._extra_state']


        else:

            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_y.weight'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_y.weight']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_y.bias'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_y.bias']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_x.weight'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_x.weight']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_x.bias'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_x.bias']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_out.weight'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_out.weight']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_out.bias'] = dm_model_weight[f'blocks.{l}.recurrent_block.linear_out.bias']

            new_state_dict[f'model.layers.{l}.recurrent_layer.conv_1d.conv_1d.weight'] = dm_model_weight[f'blocks.{l}.recurrent_block.conv_1d.w'].unsqueeze(0).permute(2,0,1)
            new_state_dict[f'model.layers.{l}.recurrent_layer.conv_1d.conv_1d.bias'] = dm_model_weight[f'blocks.{l}.recurrent_block.conv_1d.b']

            new_state_dict[f'model.layers.{l}.recurrent_layer.rg_lru.a_param'] = dm_model_weight[f'blocks.{l}.recurrent_block.rg_lru.a_param']
            new_state_dict[f'model.layers.{l}.recurrent_layer.rg_lru.input_gate.w'] = dm_model_weight[f'blocks.{l}.recurrent_block.rg_lru.input_gate.w']
            new_state_dict[f'model.layers.{l}.recurrent_layer.rg_lru.input_gate.b'] = dm_model_weight[f'blocks.{l}.recurrent_block.rg_lru.input_gate.b']
            new_state_dict[f'model.layers.{l}.recurrent_layer.rg_lru.a_gate.w'] = dm_model_weight[f'blocks.{l}.recurrent_block.rg_lru.a_gate.w']
            new_state_dict[f'model.layers.{l}.recurrent_layer.rg_lru.a_gate.b'] = dm_model_weight[f'blocks.{l}.recurrent_block.rg_lru.a_gate.b']
            
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_y._extra_state'] = model.state_dict()[f'model.layers.{l}.recurrent_layer.linear_y._extra_state']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_x._extra_state'] = model.state_dict()[f'model.layers.{l}.recurrent_layer.linear_x._extra_state']
            new_state_dict[f'model.layers.{l}.recurrent_layer.linear_out._extra_state'] = model.state_dict()[f'model.layers.{l}.recurrent_layer.linear_out._extra_state']

    model.load_state_dict(new_state_dict, strict=True)
    model = model.half()

    print("Restored!")
    model.save_to(args.output_path)
    logging.info(f'Griffin NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    args = get_args()
    convert(args)