import torch.multiprocessing as mp
from megatron.core import dist_checkpointing, parallel_state
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed

mp.set_start_method("spawn", force=True)

r"""
Conversion script to convert Megatron-LM Distributed Checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
     megatron_ckpt_to_nemo.py \
     ++checkpoint_folder=<path_to_megatronlm_dist_checkpoints_folder> \
     ++nemo_file_path=<path_to_output_nemo_file> \
     model.tensor_model_parallel_size=<tensor_model_parallel_size> \
     model.pipeline_model_parallel_size=<pipeline_model_parallel_size>
"""

# We don't want to update all the matching keys
MATCHING_KEYS_TO_UPDATE = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "kv_channels",
    "encoder_seq_length",
    "max_position_embeddings",
    "attention_dropout",
    "hidden_dropout",
    "apply_query_key_layer_scaling",
    "normalization",
    "make_vocab_size_divisible_by",
    "openai_gelu",
    "position_embedding_type",
    "rotary_base",
    "rotary_interleaved",
    "num_query_groups",
    "fp32_residual_connection",
    "bias_dropout_fusion",
    "add_position_embedding",
    "micro_batch_size",
    "global_batch_size",
    "bias_gelu_fusion",
    "bias_dropout_fusion",
    "apply_rope_fusion",
    "seed",
)

MEGATRON_KEYS2NEMO_KEYS = {
    "norm_epsilon": "layernorm_epsilon",
    "rotary_seq_len_interpolation_factor": "seq_len_interpolation_factor",
    "add_bias_linear": "bias",
    "add_qkv_bias": "qkv_bias",
    "rotary_percent": "rotary_percentage",
    "padded_vocab_size": "override_vocab_size",
}

UNMATCHED_MEGATRON_KEYS = {
    "encoder_num_layers",
    "decoder_num_layers",
    "use_rotary_position_embeddings",
    "apply_residual_connection_post_layernorm",  # Nemo don't use this
}


def update_config_with_args(cfg, args):
    vargs = vars(args)

    assert vargs['use_mcore_models'] is True, 'Megatron-LM model needs to use mcore'
    with open_dict(cfg):
        # matching keys first
        for key in MATCHING_KEYS_TO_UPDATE:
            cfg.model[key] = vargs[key]

        # Checkpoint
        cfg.restore_from_path = None
        cfg.model.dist_ckpt_format = "zarr"
        cfg.model.dist_ckpt_load_on_device = True

        # Activation
        if vargs['swiglu'] is True:
            cfg.model.activation = 'fast-swiglu'
        elif vargs['squared_relu'] is True:
            cfg.model.activation = 'squared-relu'
        elif vargs['openai_gelu'] is True:
            cfg.model.activation = 'geglu'
        else:
            logging.info('Unknown activation function. Use gelu by default.')
            cfg.model.activation = 'gelu'

        # Model configuration
        # hardcoded params
        cfg.model.mcore_gpt = True
        cfg.model.do_layer_norm_weight_decay = False
        cfg.model.pre_process = True
        cfg.model.post_process = True
        cfg.model.persist_layer_norm = True
        cfg.model.transformer_block_type = "pre_ln"
        cfg.model.normalize_attention_scores = True
        cfg.model.share_embeddings_and_output_weights = not vargs['untie_embeddings_and_output_weights']

        # update model definition using non-matfching keys between mcore and nemo
        cfg.model.transformer_engine = True if vargs['transformer_impl'] == "transformer_engine" else False
        if args.apply_layernorm_1p is True:
            cfg.model.normalization = 'layernorm1p'
        for megatron_key in MEGATRON_KEYS2NEMO_KEYS.keys():
            nemo_key = MEGATRON_KEYS2NEMO_KEYS[megatron_key]
            cfg.model[nemo_key] = vargs[megatron_key]

        # Tokenizer configuration
        # There is no tokenizer info in the megatron-lm checkpoints,
        # therefore tokenizer needs to be modified for each converted model
        # examples below:
        # cfg.model.tokenizer.library = "megatron"
        # cfg.model.tokenizer.type = "GPT2BPETokenizer"
        # cfg.model.tokenizer.vocab_file = None
        # cfg.model.tokenizer.merge_file = None

        # Trainer configuration
        cfg.trainer.precision = "bf16" if args.bf16 else "16"

        # Data configuration
        # We don't need data section for nemo checkpoint
        cfg.model.data.data_impl = "mock"
        cfg.model.data.data_prefix = None

        # Optimizer configuration
        # We don't need optimizer section for nemo checkpoint

    logging.info("\n\n************** Converted NeMo Configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    return cfg


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    logging.info('Loading Megatron-LM model..')
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

    del checkpoint['args']
    del checkpoint['checkpoint_version']
    del checkpoint['iteration']
    del checkpoint['optimizer']
    del checkpoint['opt_param_scheduler']
    del checkpoint['num_floating_point_operations_so_far']

    gpt_model.load_state_dict(checkpoint)

    logging.info('Megatron-LM model loaded correctly.')
    return gpt_model


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    checkpoint_path = cfg.checkpoint_path
    common_sd = dist_checkpointing.load_common_state_dict(checkpoint_path)
    args = common_sd['args']

    cfg = update_config_with_args(cfg, args)

    # Initialize AppState for Loading Megatron-LM model
    local_rank, rank, world_size = initialize_distributed(args)
    app_state = AppState()
    initialize_model_parallel_for_nemo(
        world_size=world_size,
        global_rank=rank,
        local_rank=local_rank,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_split_rank=0,
        micro_batch_size=None,
        global_batch_size=None,
        seed=None,
        apex_transformer_log_level=30,
    )
    app_state.data_parallel_rank = 0

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
            pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=0,
        )

    model = MegatronGPTModel(cfg.model, trainer)

    gpt_model = model.model
    load_distributed_checkpoint(checkpoint_path, gpt_model)

    # verify tensor parallel rank id and pipeline parallel rank id matches
    assert app_state.data_parallel_size == 1
    model._save_restore_connector = NLPSaveRestoreConnector()
    model.save_to(cfg.nemo_file_path)
    logging.info(f'NeMo model saved to: {cfg.nemo_file_path}')


if __name__ == '__main__':
    main()
