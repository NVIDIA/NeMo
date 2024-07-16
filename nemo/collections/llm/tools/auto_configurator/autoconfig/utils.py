# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Utility functions for the HP tool."""
import copy
from typing import List, Optional, Tuple

from nemo.collections.llm.tools.auto_configurator import base_configs

MODULES = {
    "gpt3": "GPT",
    "llama": "Llama",
    "mixtral": "Mixtral",
    "mistral": "Mistral",
}


def _calculate_model_size(
    vocab_size: int = None,
    seq_length: int = None,
    hidden_size: int = None,
    num_layers: int = None,
    ffn_size: int = None,
    kv_channels: int = None,
    att_heads: int = None,
    model_name: str = "gpt3",
):
    """
    Calculates the model size (number of parameters in billions), given the model parameters
    and name.
    :param int vocab_size: vocabulary size to be used during training.
    :param int seq_length: input sequence length to be used during training.
    :param int hidden_size: size of the hidden layers of the model.
    :param int num_layers: number of layers in the model.
    :param int ffn_size: FFN size of the model.
    :param int kv_channels: number of KV channels in the transformer layers.
    :param int att_heads: number of attention heads in the transformer layers.
    :param str model_name: name of the model, i.e gpt3, t5, mt5...
    :return: size of the model in billions of parameters.
    :rtype: float
    :raises NotImplementedError: if the model name is not valid.
    """
    if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"]:
        model_size = (
            12
            * num_layers
            * hidden_size**2
            * (1 + (13 / (12 * hidden_size)) + ((vocab_size + seq_length) / (12 * num_layers * hidden_size)))
            / 1e9
        )
    elif model_name in ["t5", "mt5"]:
        # 2 L F + 3 L P + H (2 + 4 L F + L (21 + 12 P) + 1 S + 1 V)
        proj_size = att_heads * kv_channels
        model_size = (
            2 * num_layers * 1.5 * ffn_size
            + 3 * num_layers * proj_size
            + hidden_size
            * (2 + 4 * num_layers * 1.5 * ffn_size + num_layers * (21 + 12 * proj_size) + seq_length + vocab_size)
        ) / 1e9
    elif model_name == "bert":
        model_size = (
            num_layers * (ffn_size + hidden_size * (4 * hidden_size + 3 * att_heads + 2 * ffn_size + 6))
            + hidden_size * (vocab_size + seq_length + hidden_size + 5)
        ) / 1e9

    else:
        raise NotImplementedError("Model name is not valid.")

    return model_size


def calculate_model_size_params(
    model_size_in_b: float,
    vocab_size: int = 51200,
    seq_length: int = 2048,
    model_name: str = "gpt3",
) -> Tuple[int, int, float]:
    """
    Calculates the parameters that affect model_size: hidden size, attention heads,
    KV channels, and FFN size. It also calculates the learning rate.
    :param float model_size_in_b: float, number of parameters in the desired model config, in billions.
    :param int seq_length: int, sequence length to be used during training.
    :param int vocab_size: int, size of the vocabulary to use for training.
    :param str model_name: str, name of the model to be trained, i.e. gpt3, t5, mt5...
    :returns: tuple (layers, hs, att_h, ffn, kv, lr)
        WHERE
        int layers is the number of layers in the model.
        int hs is the hidden size of the model.
        int att_h is the number of attention heads in the model.
        int ffn is the FFN hidden size of the model.
        int kv is the number of KV channels in the model.
        float lr is the learning rate used to train the model.
    :raises ValueError: if the model size is larger than the max supported model size.
    :raises NotImplementedError: if the model name is not supported.
    """
    ffn, kv = None, None  # Only needed for some models.
    if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"]:
        if model_size_in_b < 0.25:
            hs, att_h, lr = 768, 12, 6e-4
        elif model_size_in_b < 0.5:
            hs, att_h, lr = 1024, 16, 3e-4
        elif model_size_in_b < 1:
            hs, att_h, lr = 1536, 16, 2.5e-4
        elif model_size_in_b < 2:
            hs, att_h, lr = 2048, 16, 2e-4
        elif model_size_in_b < 3:
            hs, att_h, lr = 2560, 32, 1.6e-4
        elif model_size_in_b < 4.5:
            hs, att_h, lr = 3072, 32, 1.4e-4
        elif model_size_in_b < 8:
            hs, att_h, lr = 4096, 32, 1.2e-4
        elif model_size_in_b < 15:
            hs, att_h, lr = 5120, 40, 1e-4
        elif model_size_in_b < 25:
            hs, att_h, lr = 6144, 48, 1e-4
        elif model_size_in_b < 52:
            hs, att_h, lr = 8192, 64, 0.8e-4
        elif model_size_in_b < 105:
            hs, att_h, lr = 10240, 80, 0.7e-4
        elif model_size_in_b < 205:
            hs, att_h, lr = 12288, 96, 0.6e-4
        elif model_size_in_b < 405:
            hs, att_h, lr = 20480, 128, 0.5e-4
        elif model_size_in_b < 805:
            hs, att_h, lr = 20480, 128, 0.4e-4
        elif model_size_in_b < 1105:
            hs, att_h, lr = 25600, 160, 0.3e-4
        else:
            raise ValueError("Model_size for GPT-3 must be smaller than 1.1T parameters.")
    elif model_name == "t5":
        kv, lr = 64, 1e-4
        if model_size_in_b < 0.1:
            hs, att_h, ffn = 512, 6, 1024
        elif model_size_in_b < 0.4:
            hs, att_h, ffn = 768, 12, 2048
        elif model_size_in_b < 1:
            hs, att_h, ffn = 1024, 16, 2816
        elif model_size_in_b < 5:
            hs, att_h, ffn = 2048, 32, 5120
        elif model_size_in_b < 15:
            hs, att_h, ffn = 4096, 64, 10240
        elif model_size_in_b < 25.9:
            hs, att_h, ffn = 5120, 80, 10880
        elif model_size_in_b < 43.0:
            hs, att_h, ffn = 6144, 96, 10880
        elif model_size_in_b <= 85.5:
            hs, att_h, ffn = 6144, 96, 16384
        elif model_size_in_b <= 165.5:
            hs, att_h, ffn, kv = 7680, 96, 20480, 128
        elif model_size_in_b <= 250:
            hs, att_h, ffn, kv = 12288, 96, 32768, 128
        else:
            raise ValueError("Model_size for T5 must be smaller than 250B parameters.")
    elif model_name == "mt5":
        kv, lr = 64, 1e-4
        if model_size_in_b < 0.25:
            hs, att_h, ffn = 512, 6, 1024
        elif model_size_in_b < 0.5:
            hs, att_h, ffn = 768, 12, 2048
        elif model_size_in_b < 1.2:
            hs, att_h, ffn = 1024, 16, 2816
        elif model_size_in_b < 5:
            hs, att_h, ffn = 2048, 32, 5120
        elif model_size_in_b < 15:
            hs, att_h, ffn = 4096, 64, 10240
        elif model_size_in_b < 25.9:
            hs, att_h, ffn = 5120, 80, 10880
        elif model_size_in_b < 43.0:
            hs, att_h, ffn = 6144, 96, 10880
        elif model_size_in_b <= 85.5:
            hs, att_h, ffn = 6144, 96, 16384
        elif model_size_in_b <= 165.5:
            hs, att_h, ffn, kv = 7680, 96, 20480, 128
        elif model_size_in_b <= 250:
            hs, att_h, ffn, kv = 12288, 96, 32768, 128
        else:
            raise ValueError("Model_size for mT5 must be smaller than 250B parameters.")
    elif model_name == "bert":
        lr = 1e-4
        if model_size_in_b < 0.25:
            hs, att_h, lr = 768, 12, 2e-4
        elif model_size_in_b < 0.5:
            hs, att_h, lr = 1024, 16, 2e-4
        elif model_size_in_b < 1:
            hs, att_h = 1536, 16
        elif model_size_in_b < 2:
            hs, att_h = 2048, 16
        elif model_size_in_b < 3:
            hs, att_h = 2560, 32
        elif model_size_in_b < 4.5:
            hs, att_h = 2560, 32
        elif model_size_in_b < 8:
            hs, att_h = 4096, 32
        elif model_size_in_b < 15:
            hs, att_h = 5120, 40
        elif model_size_in_b <= 25:
            hs, att_h = 6144, 48
        elif model_size_in_b <= 46.5:
            hs, att_h = 7680, 48
        elif model_size_in_b <= 87.5:
            hs, att_h = 9216, 96
        elif model_size_in_b <= 165.5:
            hs, att_h = 9216, 96
        elif model_size_in_b <= 250.5:
            hs, att_h = 12288, 96
        else:
            raise ValueError("Model_size for BERT must be smaller than 25B parameters.")
        ffn = 4 * hs
    else:
        raise NotImplementedError("Model name is not valid.")

    # Try powers of 2
    margin = 0.01
    for attempt in range(0, 10):
        for layers in (2**p for p in range(1, 10)):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 16
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(16, 201, 16):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 2
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(2, 201, 2):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try multiples of 5
    margin = 0.01
    for attempt in range(0, 6):
        for layers in range(5, 201, 5):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.

    # Try any valid number
    margin = 0.01
    for attempt in range(0, 10):
        for layers in range(1, 200):
            out_size = _calculate_model_size(
                vocab_size=vocab_size,
                seq_length=seq_length,
                hidden_size=hs,
                num_layers=layers,
                ffn_size=ffn,
                kv_channels=kv,
                att_heads=att_h,
                model_name=model_name,
            )
            if model_size_in_b * (1.0 - margin) < out_size < model_size_in_b * (1.0 + margin):
                return layers, hs, att_h, ffn, kv, lr
        margin += 0.01  # Double margin of acceptable model sizes.
    raise Exception("Number of layers not found, config is not possible.")


def generic_base_config(
    model_name: str = "llama",
    model_version: int = 2,
    model_size_in_b: int = 7,
    model_measure: str = "B",
    cfg: dict = {},
) -> dict:
    """
    Generates a base config dictionary from a base config yaml file.
    :param omegaconf.dictconfig.DictConfig cfg: hydra-like config object for the HP tool.
    :param str model_name: name of the model, i.e. gpt3, t5, mt5...
    :returns: dictionary containing the base configuration for the model.
    :rtype: dict
    """

    model_cls = getattr(base_configs, MODULES[model_name])
    model = model_cls(version=model_version, size=model_size_in_b, measure=model_measure, cfg=cfg)

    base_cfg = {
        "model": model.get_model_config(),
        "optim": model.get_optim_config(),
        "tokenizer": model.get_tokenizer_config(),
        "trainer": model.get_trainer_config(),
        "data": model.get_data_config(),
        "run": model.get_run_config(),
    }

    return base_cfg


def modify_cfg(
    base_cfg: dict,
    act: int,
    num_mbs_act: int,
    act_per_pipe: int,
    tp: int,
    pp: int,
    cp: int,
    ep: int,
    virtual_pipelines: int,
    mbs: int,
    max_minutes: int,
    max_steps: int,
    num_nodes: int,
    model_name: str,
) -> dict:
    """
    Modify the base configuration for the model with the new parameters that are specific to the current model, which the HP tool heuristics selected.
    :param dict base_cfg: base configuration for the current model, which will be modified in this function.
    :param int act: number of activation checkpointing layers to use for the model.
    :param int num_mbs_act:
    :param int act_per_pipe:
    :param int tp: Tensor Parallelism (TP) value to be set for the model.
    :param int pp: Pipeline Parallelism (PP) value to be set for the model.
    :param int cp: Context Parallelism (CP) value to be set for the model.
    :param int ep: Expert Parallelism (EP) value to be set for the model.
    :param int virtual_pipelines: Virtual Pipelines value to be set for the model.
    :param int mbs: Micro Batch Size (MBS) value to be set for the model.
    :param int max_minutes: maximum amount of time to run this model for.
    :param int max_steps: maximum number of steps to run this model for.
    :param int num_nodes: number of nodes to use for the training run.
    :param str model_name: name of the model, i.e. gpt3, t5, mt5...
    :return: dictionary containing the updated model configuration parameters.
    :rtype: dict
    """
    new_cfg = copy.deepcopy(base_cfg)
    if act is not None:
        if model_name in [
            "gpt3",
            "bert",
            "llama",
            "baichuan2",
            "chatglm",
            "qwen2",
            "mixtral",
            "mistral",
        ]:
            new_cfg["auto_config"]["activations_checkpoint_num_layers"] = act
        else:
            new_cfg["auto_config"]["encoder"]["activations_checkpoint_num_layers"] = act // 2
            new_cfg["auto_config"]["decoder"]["activations_checkpoint_num_layers"] = act // 2

    if num_mbs_act is not None and model_name in [
        "gpt3",
        "bert",
        "llama",
        "baichuan2",
        "chatglm",
        "qwen2",
        "mixtral",
        "mistral",
    ]:
        new_cfg["auto_config"]["num_micro_batches_with_partial_activation_checkpoints"] = num_mbs_act

    if act_per_pipe is not None and model_name in [
        "gpt3",
        "bert",
        "llama",
        "baichuan2",
        "chatglm",
        "qwen2",
        "mixtral",
        "mistral",
    ]:
        new_cfg["auto_config"]["activations_checkpoint_layers_per_pipeline"] = act_per_pipe

    if virtual_pipelines is not None and model_name in [
        "gpt3",
        "bert",
        "llama",
        "baichuan2",
        "chatglm",
        "qwen2",
        "mixtral",
        "mistral",
    ]:
        new_cfg["auto_config"]["virtual_pipeline_model_parallel_size"] = virtual_pipelines

    new_cfg["auto_config"]["tensor_model_parallel_size"] = tp
    new_cfg["auto_config"]["pipeline_model_parallel_size"] = pp
    new_cfg["auto_config"]["micro_batch_size"] = mbs

    if cp is not None:
        new_cfg["auto_config"]["context_parallel_size"] = cp

    if ep is not None:
        new_cfg["auto_config"]["expert_model_parallel_size"] = ep

    if model_name in [
        "gpt3",
        "bert",
        "llama",
        "baichuan2",
        "chatglm",
        "qwen2",
        "mixtral",
        "mistral",
    ]:
        att_heads = new_cfg["model"].num_attention_heads
        num_layers = new_cfg["model"].num_layers
    else:
        att_heads = new_cfg["model"].encoder.num_attention_heads
        num_layers = new_cfg["model"].encoder.num_layers

    # gbs = mbs * num_gpus * accumulate_grad_batches / (tp * pp)
    num_gpus = new_cfg["trainer"]["num_nodes"] * new_cfg["trainer"]["devices"]
    gbs = new_cfg["model"].global_batch_size
    seq_len = new_cfg["model"].encoder_seq_length

    mod_gbs = gbs % (mbs * num_gpus / (tp * pp))
    mod_att_heads = att_heads % tp
    mod_layers = num_layers % pp
    if mod_gbs == 0 and mod_att_heads == 0 and mod_layers == 0:
        # Valid config
        new_cfg["trainer"]["num_nodes"] = num_nodes  # Necessary for short single-node test.
        new_cfg["trainer"]["max_steps"] = max_steps
        new_cfg["trainer"]["val_check_interval"] = max_steps
        days = max_minutes // 3600
        hours = (max_minutes % 3600) // 60
        mins = (max_minutes % 3600) % 60
        new_cfg["run"]["time_limit"] = f"{days}-{hours}:{mins}:00"
        new_cfg["run"][
            "name"
        ] = f"{new_cfg['run']['name']}_{num_nodes}nodes_tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_mbs_{mbs}_act_ckpt_{act}_num_mbs_act_{num_mbs_act}_act_per_pipe_{act_per_pipe}"
        print(
            f"Valid config: SeqLen={seq_len}, GBS={gbs}, MBS={mbs}, TP={tp}, PP={pp}, CP={cp}, EP={ep}, act_ckpt_layers={act}, num_mbs_act={num_mbs_act}, act_per_pipe={act_per_pipe}. Adding to directory."
        )
        return new_cfg
    return None
