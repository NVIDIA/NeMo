# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import AppState, logging

r"""
Script to convert a legacy (non-mcore path) nemo checkpoint into mcore-path checkpoint for GPT models.

*Important* Before running this script, please first
1) convert your legacy checkpoint to TP1 PP1 format:
    python examples/nlp/language_modeling/megatron_change_num_partitions.py \
    <follow the readme in that script> \
    --target_tensor_model_parallel_size=1 \
    --target_pipeline_model_parallel_size=1
2) extract your checkpoint to a folder with
    tar -xvf your_ckpt.nemo
        
Then, run this conversion script:
python convert_nemo_gpt_to_mcore.py \
 --in-file <path to extracted, TP1 PP1 legacy checkpoint folder> \
 --out-file <path to output nemo ile>
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to extracted, TP1 PP1 NeMo GPT checkpoint.",
    )
    parser.add_argument(
        "--out-file", type=str, default=None, required=True, help="Path to output mcore weights file (ends in .nemo)."
    )
    args = parser.parse_args()
    return args


def get_mcore_model_from_nemo_ckpt(nemo_restore_from_path):
    model_cfg = MegatronGPTModel.restore_from(nemo_restore_from_path, return_config=True)
    model_cfg.tokenizer.vocab_file = None
    model_cfg.tokenizer.merge_file = None
    model_cfg.mcore_gpt = True

    logging.info("*** initializing mcore model with the following config")
    logging.info(OmegaConf.to_yaml(model_cfg))
    trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())

    app_state = AppState()
    if os.path.isdir(nemo_restore_from_path):
        app_state.nemo_file_folder = nemo_restore_from_path
    else:
        logging.warning(
            "`nemo_file_folder` is NOT set because checkpoint is not pre-extracted. Subsequent operations may fail."
        )
    mcore_model = MegatronGPTModel(model_cfg, trainer=trainer)
    return mcore_model


def print_mcore_parameter_names(restore_from_path):
    mcore_model = get_mcore_model_from_nemo_ckpt(restore_from_path)

    print("*********")
    print('\n'.join(sorted([k + '###' + str(v.shape) for k, v in mcore_model.named_parameters()])))
    print("*********")


def build_key_mapping(nemo_cfg, use_O2_prefix=None):
    num_layers = nemo_cfg.num_layers
    has_bias = nemo_cfg.get("bias", True)
    if use_O2_prefix is None:
        use_O2_prefix = nemo_cfg.get('megatron_amp_O2', False)
    model_str = 'model.module' if use_O2_prefix else 'model'

    # For GPT there is a 1:1 mapping of keys
    mcore_to_nemo_mapping = {
        f"{model_str}.embedding.word_embeddings.weight": "model.language_model.embedding.word_embeddings.weight",
        f"{model_str}.decoder.final_layernorm.bias": "model.language_model.encoder.final_layernorm.bias",
        f"{model_str}.decoder.final_layernorm.weight": "model.language_model.encoder.final_layernorm.weight",
    }
    if not nemo_cfg.get("share_embeddings_and_output_weights", True):
        mcore_to_nemo_mapping[f"{model_str}.output_layer.weight"] = "model.language_model.output_layer.weight"

    if nemo_cfg.get("position_embedding_type", 'learned_absolute') == 'rope':
        mcore_to_nemo_mapping[f"{model_str}.rotary_pos_emb.inv_freq"] = "model.language_model.rotary_pos_emb.inv_freq"
    else:
        mcore_to_nemo_mapping[
            f"{model_str}.embedding.position_embeddings.weight"
        ] = "model.language_model.embedding.position_embeddings.weight"

    nemo_prefix = "model.language_model.encoder.layers"
    mcore_prefix = f"{model_str}.decoder.layers"
    for i in range(num_layers):
        for wb in ('weight', 'bias') if has_bias else ('weight',):
            mcore_to_nemo_mapping.update(
                {
                    f"{mcore_prefix}.{i}.mlp.linear_fc2.{wb}": f"{nemo_prefix}.{i}.mlp.dense_4h_to_h.{wb}",
                    f"{mcore_prefix}.{i}.mlp.linear_fc1.{wb}": f"{nemo_prefix}.{i}.mlp.dense_h_to_4h.{wb}",
                    f"{mcore_prefix}.{i}.self_attention.linear_proj.{wb}": f"{nemo_prefix}.{i}.self_attention.dense.{wb}",
                    f"{mcore_prefix}.{i}.self_attention.linear_qkv.{wb}": f"{nemo_prefix}.{i}.self_attention.query_key_value.{wb}",
                }
            )
        # layernorm layers always have bias!
        for wb in ('weight', 'bias'):
            mcore_to_nemo_mapping.update(
                {
                    f"{mcore_prefix}.{i}.self_attention.linear_qkv.layer_norm_{wb}": f"{nemo_prefix}.{i}.input_layernorm.{wb}",
                    f"{mcore_prefix}.{i}.mlp.linear_fc1.layer_norm_{wb}": f"{nemo_prefix}.{i}.post_attention_layernorm.{wb}",
                }
            )

    return mcore_to_nemo_mapping


def load_model(model, state_dict):
    # try:
    for name, module in model.named_parameters():
        if name in state_dict:
            module.data = state_dict.pop(name)
        else:
            raise RuntimeError(f"Unexpected key: {name} not in state_dict but in model.")

    for name, buffer in model.named_buffers():
        if name in state_dict:
            buffer.data = state_dict.pop(name)

    if len(state_dict.keys()) != 0:
        raise RuntimeError(f"Additional keys: {state_dict.keys()} in state_dict but not in model.")

    return model


def convert(input_ckpt_file, output_ckpt_file, skip_if_output_exists=True):
    if skip_if_output_exists and os.path.exists(output_ckpt_file):
        logging.info(f"Output file already exists ({output_ckpt_file}), skipping conversion...")
        return
    dummy_trainer = Trainer(devices=1, accelerator='cpu')

    nemo_model = MegatronGPTModel.restore_from(input_ckpt_file, trainer=dummy_trainer)
    nemo_tokenizer_model = nemo_model.cfg.tokenizer.model
    nemo_state_dict = nemo_model.state_dict()
    mcore_state_dict = OrderedDict()
    for mcore_param, nemo_param in build_key_mapping(nemo_model.cfg).items():
        mcore_state_dict[mcore_param] = nemo_state_dict[nemo_param]

    mcore_model = get_mcore_model_from_nemo_ckpt(input_ckpt_file)
    mcore_model = load_model(mcore_model, mcore_state_dict)

    if nemo_model.cfg.tokenizer.model is not None:
        logging.info("registering artifact: tokenizer.model = " + nemo_tokenizer_model)
        mcore_model.register_artifact("tokenizer.model", nemo_tokenizer_model)

    mcore_model.save_to(output_ckpt_file)
    logging.info(f"Done. Model saved to {output_ckpt_file}")


def run_sanity_checks(nemo_ckpt_file, mcore_ckpt_file):
    cfg = OmegaConf.load(
        os.path.join(
            os.path.dirname(__file__),
            '../../examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_tuning_config.yaml',
        )
    )

    cfg.trainer.precision = 'bf16'  # change me
    dtype = torch.bfloat16
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    nemo_model = MegatronGPTModel.restore_from(nemo_ckpt_file, trainer=trainer).eval().to(dtype)
    mcore_model = MegatronGPTModel.restore_from(mcore_ckpt_file, trainer=trainer).eval().to(dtype)

    logging.debug("*** Mcore model restored config")
    logging.debug(OmegaConf.to_yaml(mcore_model.cfg))

    nemo_summary = nemo_model.summarize()
    mcore_summary = mcore_model.summarize()

    logging.info("Sanity checks:")

    # check num weights match
    assert nemo_summary.total_parameters == mcore_summary.total_parameters, "❌ total parameters do not match"
    assert nemo_summary.model_size == mcore_summary.model_size, "❌ model sizes do not match"
    logging.info("✅ Number of weights match")

    # check weights match
    mcore_state_dict = mcore_model.state_dict()
    nemo_state_dict = nemo_model.state_dict()
    for mcore_param, nemo_param in build_key_mapping(nemo_model.cfg, use_O2_prefix=False).items():
        try:
            assert torch.allclose(
                mcore_state_dict[mcore_param], nemo_state_dict[nemo_param]
            ), f"❌ parameter {mcore_param} does not match"
        except KeyError:
            buffers = [k for k, v in mcore_model.named_buffers()]
            assert (
                mcore_param in buffers or mcore_param.replace('model.', 'model.module.', 1) in buffers
            ), f"❌ parameter {mcore_param} is not found in the state dict or named_buffers()"
    logging.info("✅ Weights match")


if __name__ == '__main__':
    args = get_args()

    input_ckpt = args.in_file
    output_ckpt = args.out_file
    os.makedirs(os.path.dirname(output_ckpt), exist_ok=True)
    convert(input_ckpt, output_ckpt, skip_if_output_exists=True)
    torch.cuda.empty_cache()
    run_sanity_checks(input_ckpt, output_ckpt)
