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

"""Prepares and launches the training HP search using nemo_framework_launcher."""

import os
import shutil
import subprocess
from typing import List, Tuple

from autoconfig import utils


def search_training_config(
    base_cfg: dict,
    train_cfg: dict,
) -> None:
    """
    Entry point for the training HP search. This function calls other functions to perform three
    actions: generates the grid of possible configurations; launches those configurations using
    nemo_framework_launcher; and launches a final job to compare the results of all the training
    jobs.
    :param dict base_cfg: base configuration of the model to be trained.
    :param float model_size_in_b: number of parameters in the model, if known.
    :param str model_name: name of the model to be trained: gpt3, t5, mt5...
    :param str hydra_args: hydra override arguments in string format.
    :param omegaconf.dictconfig.fDictConfig cfg: main hydra config object for the HP tool.
    :return: None
    """
    # Generate candidate configs.
    configs = generate_grid_search_configs(base_cfg, train_cfg)

    return configs


def generate_grid_search_configs(
    base_cfg: dict,
    train_cfg: dict,
) -> Tuple[str, List[int], int]:
    """
    Generates the grid of all possible configurations for the given model, and stores
    each different configuration in a yaml file.
    :param dict base_cfg: base configuration of the model to be trained.
    :param float model_size_in_b: number of parameters in the model.
    :param str model_name: name of the model to be trained: gpt3, t5, mt5...
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object for the HP tool.
    :returns: tuple (base_dir, results_cfgs, num_nodes)
        WHERE
        str base_dir is the path to the directory where the results will be stored.
        List[int] results_cfgs is a list of all the config names that were generated.
        int num_nodes is the number of nodes used to run each config.
    """

    model_name = train_cfg["model_type"]
    model_version = train_cfg["model_version"]
    model_size_in_b = train_cfg["model_size_in_b"]
    model_measure = train_cfg["model_measure"]

    # 2 * num_layers is needed because of encoder/decoder architecture.
    multiplier = (
        1 if model_name in ["gpt3", "bert", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"] else 2
    )

    seq_length = base_cfg["model"].seq_length
    num_layers = (
        base_cfg["model"].num_layers
        if model_name in ["gpt3", "bert", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"]
        else base_cfg["model"].encoder.num_layers
    )

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
        act_method = base_cfg["model"].activations_checkpoint_method
    else:
        act_method = base_cfg["model"].encoder.activations_checkpoint_method

    (
        tp_list,
        pp_list,
        cp_list,
        ep_list,
        mbs_list,
        min_model_parallel,
        max_model_parallel,
        gbs,
    ) = _calculate_tp_pp_mbs_grid(
        model_size_in_b=model_size_in_b,
        num_layers=num_layers,
        model_name=model_name,
        seq_length=seq_length,
        train_cfg=train_cfg,
    )

    max_minutes = train_cfg.get("max_minutes_per_run")
    max_steps = train_cfg.get("max_steps_per_run")
    num_nodes = train_cfg.get("num_nodes")

    valid_tp_pp_list = []
    for tp in tp_list:
        for pp in pp_list:
            for cp in cp_list:
                for ep in ep_list:
                    for mbs in mbs_list:
                        num_gpus = base_cfg["trainer"]["num_nodes"] * base_cfg["trainer"]["devices"]
                        base_cfg["model"].global_batch_size = gbs
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
                            att_heads = base_cfg["model"].num_attention_heads
                            num_layers = base_cfg["model"].num_layers
                        else:
                            att_heads = base_cfg["model"].encoder.num_attention_heads
                            num_layers = base_cfg["model"].encoder.num_layers
                        model_parallelism = (tp * pp * cp * ep) if (cp and ep) else (tp * pp)
                        mod_gbs = gbs % (mbs * num_gpus / model_parallelism)
                        mod_att_heads = att_heads % tp
                        mod_layers = (multiplier * num_layers) % pp
                        mod_cp = cp if cp else 1
                        mod_ep = ep if ep else 1
                        if (
                            mod_gbs == 0
                            and mod_att_heads == 0
                            and mod_layers == 0
                            and (tp, pp, cp, ep) not in valid_tp_pp_list
                            and (mod_cp // mod_ep == mod_cp or mod_ep // mod_cp == mod_ep)
                            and min_model_parallel <= model_parallelism <= max_model_parallel
                        ):
                            valid_tp_pp_list.append((tp, pp, cp, ep))

    # Generate grid search configs.
    configs, base_cfg["auto_config"] = {}, {}
    for tp, pp, cp, ep in valid_tp_pp_list:
        (
            virtual_pipelines,
            act_ckpt_layers,
            num_micro_batches_partial_act_ckpt,
            act_ckpt_layers_per_pipeline,
        ) = _set_activations_checkpoint_params(
            tp,
            pp,
            cp,
            ep,
            num_layers,
            act_method,
            multiplier,
            model_size_in_b,
            model_name,
            model_measure,
        )
        for mbs in mbs_list:
            kwargs = {
                "base_cfg": base_cfg,
                "act": None,
                "num_mbs_act": None,
                "act_per_pipe": None,
                "tp": tp,
                "pp": pp,
                "cp": cp,
                "ep": ep,
                "virtual_pipelines": virtual_pipelines,
                "mbs": mbs,
                "max_minutes": max_minutes,
                "max_steps": max_steps,
                "num_nodes": num_nodes,
                "model_name": model_name,
            }
            if act_ckpt_layers[0] is not None:
                if act_layers is not None and act_layers != "auto":
                    act_ckpt_layers = act_layers
                for act in act_ckpt_layers:
                    for num_mbs_act in num_micro_batches_partial_act_ckpt:
                        for act_per_pipe in act_ckpt_layers_per_pipeline:
                            kwargs["act"] = act
                            kwargs["num_mbs_act"] = num_mbs_act
                            kwargs["act_per_pipe"] = act_per_pipe
                            new_cfg = utils.modify_cfg(**kwargs)
                            if new_cfg:  # Save candidate cfg.
                                configs[new_cfg["run"]["name"]] = new_cfg
            else:
                new_cfg = utils.modify_cfg(**kwargs)
                if new_cfg:  # Save candidate cfg.
                    configs[new_cfg["run"]["name"]] = new_cfg

    print(f"\nAll candidate configurations created correctly. Total number of configs: {len(configs)}.\n")
    return configs


def _set_activations_checkpoint_params(
    tp, pp, cp, ep, num_layers, act_method, multiplier, model_size_in_b, model_name, model_measure
):
    act_multiple = 4 // pp
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if act_method == "block":
        if 1.0 <= model_size_in_b < 11.3:
            act_multiple = 8 // pp
        elif 11.3 <= model_size_in_b < 26.0:
            act_multiple = 16 // pp
        elif 26.0 <= model_size_in_b < 60.0:
            act_multiple = 16 // pp
        elif 60.0 <= model_size_in_b:
            act_multiple = 32 // pp
    act_multiple = max(act_multiple, 1)

    virtual_pipelines = None
    # Num micro batches with partial act ckpt
    min_micro_b = 0  # 0 will not be used, minimum will be set to 1 later in the code.
    max_micro_b = pp
    interval_micro_b = 1
    # Act ckpt layers per pipeline
    min_layers_per_pipe = 0
    max_layers_per_pipe = num_layers
    interval_layers_per_pipe = act_multiple
    if (
        model_name in ["gpt3", "bert", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"] and pp > 2
    ):  # Interleaved pipeline scheduling.
        virtual_pipelines = num_layers // pp  # TODO: verify that this is the best value.
        act_multiple = 1
        max_micro_b = pp * (virtual_pipelines - 1) + (pp - 1) * 2 + 1
        interval_micro_b = virtual_pipelines * 8
        max_layers_per_pipe = multiplier * num_layers // pp // virtual_pipelines + 1

    (
        act_ckpt_layers,
        num_micro_batches_partial_act_ckpt,
        act_ckpt_layers_per_pipeline,
    ) = ([None], [None], [None])
    if act_method == "block":
        # Act ckpt num layers
        if virtual_pipelines is None:
            act_ckpt_layers = range(0, multiplier * num_layers // pp + 1, act_multiple)
        else:
            act_ckpt_layers = range(0, multiplier * num_layers // pp // virtual_pipelines + 1, act_multiple)

        if pp > 1 and model_name in [
            "gpt3",
            "bert",
            "llama",
            "baichuan2",
            "chatglm",
            "qwen2",
            "mixtral",
            "mistral",
        ]:
            # Num micro batches with partial act ckpt
            num_micro_batches_partial_act_ckpt = list(range(min_micro_b, max_micro_b + 1, interval_micro_b))
            if num_micro_batches_partial_act_ckpt[0] == 0:
                num_micro_batches_partial_act_ckpt[0] = 1

            # Act ckpt layers per pipeline
            act_ckpt_layers_per_pipeline = range(
                min_layers_per_pipe, max_layers_per_pipe + 1, interval_layers_per_pipe
            )

    return (
        virtual_pipelines,
        act_ckpt_layers,
        num_micro_batches_partial_act_ckpt,
        act_ckpt_layers_per_pipeline,
    )


def _tp_pp_mbs_grid_gpt3_80gb(
    model_size_in_b: float, valid_pp: List[int], seq_length: int, model_measure: str
) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for GPT-3 and 80GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [1]
    ep = [1]
    mbs = [1, 2, 3, 4, 6, 8]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1024
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if seq_length == 2048:
        if model_size_in_b <= 1.0:
            tp = [1, 2]
            gbs = 256
        elif model_size_in_b <= 4.0:
            tp = [1, 2, 4]
            gbs = 1024
        elif model_size_in_b <= 8.0:
            tp = [1, 2, 4]
            gbs = 2048
        elif model_size_in_b <= 13.0:
            tp = [1, 2, 4, 8]
            gbs = 2048
        elif model_size_in_b <= 23.0:
            tp = [1, 2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 4]
            mbs = [1, 2, 4]
            min_model_parallel = 4
            max_model_parallel = 8
            gbs = 2048
        elif model_size_in_b <= 45.0:
            tp = [2, 4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 4]
            mbs = [1, 2, 4]
            min_model_parallel = 8
            max_model_parallel = 32
            gbs = 2048
        elif model_size_in_b <= 95:
            tp = [2, 4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 8]
            mbs = [1, 2, 4, 8]
            min_model_parallel = 8
            max_model_parallel = 64
            gbs = 2048
        elif model_size_in_b <= 130.0:
            tp = [2, 4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 16]
            mbs = [1, 2, 4, 8]
            min_model_parallel = 16
            max_model_parallel = 128
            gbs = 2048
        elif model_size_in_b <= 195.0:
            tp = [8]
            pp = [x for x in valid_pp if 4 <= x <= 16]
            mbs = [1, 2, 4]
            min_model_parallel = 32
            max_model_parallel = 256
            gbs = 2048
        elif model_size_in_b <= 395.0:
            tp = [8]
            pp = [x for x in valid_pp if 8 <= x <= 32]
            mbs = [1, 2, 4]
            min_model_parallel = 64
            max_model_parallel = 512
            gbs = 2048
        elif model_size_in_b <= 790.0:
            tp = [8]
            pp = [x for x in valid_pp if 8 <= x <= 100]
            mbs = [1, 2, 4]
            min_model_parallel = 128
            max_model_parallel = 1024
            gbs = 2048
        elif model_size_in_b <= 1100.0:
            tp = [8]
            pp = [x for x in valid_pp if 16 <= x <= 130]
            mbs = [1, 2, 4]
            min_model_parallel = 256
            max_model_parallel = 2048
            gbs = 2048
    elif seq_length == 4096:
        if model_size_in_b <= 1.0:
            tp = [1, 2, 4]
            mbs = [1, 2, 4, 8]
            gbs = 128
        elif model_size_in_b <= 4.0:
            tp = [1, 2, 4]
            mbs = [1, 2, 4, 8]
            gbs = 512
        elif model_size_in_b <= 8.0:
            tp = [1, 2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2, 4]
            gbs = 1024
        elif model_size_in_b <= 13.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2, 4]
            gbs = 1024
        elif model_size_in_b <= 23.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2]
            min_model_parallel = 4
            max_model_parallel = 16
            gbs = 1024
        elif model_size_in_b <= 45.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 2 <= x <= 4]
            mbs = [1, 2]
            min_model_parallel = 8
            max_model_parallel = 32
            gbs = 1024
        elif model_size_in_b <= 95:
            tp = [4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 8]
            mbs = [1, 2]
            min_model_parallel = 8
            max_model_parallel = 64
            gbs = 1024
    elif seq_length == 8192:
        if model_size_in_b <= 1.0:
            tp = [1, 2]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2, 4]
            gbs = 64
        elif model_size_in_b <= 4.0:
            tp = [1, 2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2, 4]
            gbs = 128
        elif model_size_in_b <= 8.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2]
            gbs = 256
        elif model_size_in_b <= 13.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1, 2]
            gbs = 256
        elif model_size_in_b <= 23.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 4]
            mbs = [1]
            min_model_parallel = 8
            max_model_parallel = 32
            gbs = 256
        elif model_size_in_b <= 45.0:
            tp = [8]
            pp = [x for x in valid_pp if 4 <= x <= 8]
            mbs = [1]
            min_model_parallel = 32
            max_model_parallel = 64
            gbs = 256
    elif seq_length == 16384:
        if model_size_in_b <= 1.0:
            tp = [2, 4]
            mbs = [1, 2]
            gbs = 32
        elif model_size_in_b <= 4.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1]
            gbs = 64
        elif model_size_in_b <= 8.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1]
            gbs = 128
        elif model_size_in_b <= 13.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1]
            gbs = 128
        elif model_size_in_b <= 23.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 2 <= x <= 4]
            mbs = [1]
            min_model_parallel = 8
            max_model_parallel = 32
            gbs = 128
    elif seq_length == 32768:
        if model_size_in_b <= 1.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1]
            gbs = 16
        elif model_size_in_b <= 4.0:
            tp = [2, 4]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            mbs = [1]
            gbs = 32
        elif model_size_in_b <= 8.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            min_model_parallel = 4
            max_model_parallel = 16
            mbs = [1]
            gbs = 64
        elif model_size_in_b <= 13.0:
            tp = [4, 8]
            pp = [x for x in valid_pp if 1 <= x <= 2]
            min_model_parallel = 4
            max_model_parallel = 16
            mbs = [1]
            gbs = 64
        elif model_size_in_b <= 23.0:
            tp = [8]
            pp = [x for x in valid_pp if 2 <= x <= 4]
            mbs = [1]
            min_model_parallel = 16
            max_model_parallel = 32
            gbs = 64

    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _tp_pp_mbs_grid_gpt3_40gb(model_size_in_b: float, valid_pp: List[int], model_measure: str) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for GPT-3 and 40GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [1]
    ep = [1]
    mbs = [1, 2, 4, 6, 8, 10, 12, 16]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1024
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if model_size_in_b <= 1.0:
        tp = [1, 2, 4]
        mbs = [1, 2, 4, 8]
        gbs = 256
    elif model_size_in_b <= 4.0:
        tp = [1, 2, 4, 8]
        mbs = [1, 2, 4, 8]
        gbs = 1024
    elif model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        pp = [1, 2]
        mbs = [1, 2, 4]
        min_model_parallel = 2
        gbs = 2048
    elif model_size_in_b <= 13.0:
        tp = [4, 8]
        pp = [1, 2, 4]
        mbs = [1, 2, 4]
        min_model_parallel = 4
        max_model_parallel = 32
        gbs = 2048
    elif model_size_in_b <= 23.0:
        tp = [2, 4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        min_model_parallel = 8
        max_model_parallel = 64
        gbs = 2048
    elif model_size_in_b <= 45.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 12]
        mbs = [1, 2, 4]
        min_model_parallel = 16
        max_model_parallel = 128
        gbs = 2048
    elif model_size_in_b <= 95:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 16]
        mbs = [1, 2, 4]
        min_model_parallel = 16
        max_model_parallel = 256
        gbs = 2048
    elif model_size_in_b <= 130.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 26]
        mbs = [1, 2]
        min_model_parallel = 32
        max_model_parallel = 512
        gbs = 2048
    elif model_size_in_b <= 195.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 32]
        mbs = [1, 2]
        min_model_parallel = 64
        max_model_parallel = 1024
        gbs = 2048
    elif model_size_in_b <= 395.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 4 <= x <= 64]
        mbs = [1, 2]
        min_model_parallel = 128
        max_model_parallel = 2048
        gbs = 2048
    elif model_size_in_b <= 790.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 8 <= x <= 128]
        mbs = [1, 2]
        min_model_parallel = 256
        max_model_parallel = 4096
        gbs = 2048
    elif model_size_in_b <= 1100.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 8 <= x <= 192]
        mbs = [1, 2]
        min_model_parallel = 512
        max_model_parallel = 8192
        gbs = 2048
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _tp_pp_mbs_grid_t5_80gb(model_size_in_b: float, valid_pp: List[int], model_measure: str) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for T5/mT5 and 80GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 4, 6, 8, 12, 16]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1920
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if model_size_in_b <= 1.0:
        tp = [1, 2]
        mbs = [16, 32, 64, 128]
        gbs = 2048
    elif model_size_in_b <= 4.0:
        tp = [1, 2, 4]
        mbs = [4, 6, 8, 12, 16, 24, 32, 48]
        gbs = 1920
    elif model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        mbs = [4, 6, 8, 12, 16, 24, 32]
        gbs = 1920
    elif model_size_in_b <= 14.5:
        tp = [4, 8]
        mbs = [2, 4, 6, 8, 12, 16, 24]
        gbs = 1920
    elif model_size_in_b <= 25.9:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 2]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 4
        max_model_parallel = 16
        gbs = 1920
    elif model_size_in_b <= 43.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 4]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 8
        max_model_parallel = 32
        gbs = 1920
    elif model_size_in_b <= 85.5:
        tp = [4, 8]
        pp = [x for x in valid_pp if 2 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 16
        max_model_parallel = 64
        gbs = 1920
    elif model_size_in_b <= 165.5:
        tp = [8]
        pp = [x for x in valid_pp if 4 <= x <= 16]
        mbs = [1, 2, 4, 6]
        min_model_parallel = 32
        max_model_parallel = 128
        gbs = 1920
    elif model_size_in_b <= 250:
        tp = [8]
        pp = [x for x in valid_pp if 4 <= x <= 32]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 64
        max_model_parallel = 256
        gbs = 1920
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _tp_pp_mbs_grid_t5_40gb(model_size_in_b: float, valid_pp: List[int], model_measure: str) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for T5/mT5 and 40GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    tp = [1, 2, 4, 8]
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 4, 6, 8, 12, 16]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1920
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if model_size_in_b <= 1.0:
        tp = [1, 2]
        mbs = [16, 32, 64, 128]
        gbs = 2048
    elif model_size_in_b <= 4.0:
        tp = [1, 2, 4]
        mbs = [4, 8, 12, 16, 24, 32, 48]
        gbs = 1920
    elif model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        mbs = [4, 6, 8, 12, 16, 24]
        gbs = 1920
    elif model_size_in_b <= 14.5:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 2]
        mbs = [2, 4, 6, 8, 12, 16]
        min_model_parallel = 4
        max_model_parallel = 16
        gbs = 1920
    elif model_size_in_b <= 25.9:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 8
        max_model_parallel = 32
        gbs = 1920
    elif model_size_in_b <= 43.0:
        tp = [4, 8]
        pp = [x for x in valid_pp if 1 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 16
        max_model_parallel = 32
        gbs = 1920
    elif model_size_in_b <= 85.5:
        tp = [8]
        pp = [x for x in valid_pp if 2 <= x <= 8]
        mbs = [1, 2, 4, 6, 8]
        min_model_parallel = 32
        max_model_parallel = 64
        gbs = 1920
    elif model_size_in_b <= 165.5:
        tp = [8]
        pp = [x for x in valid_pp if 4 <= x <= 32]
        mbs = [1, 2, 4]
        min_model_parallel = 64
        max_model_parallel = 128
        gbs = 1920
    elif model_size_in_b <= 250:
        tp = [8]
        pp = [x for x in valid_pp if 8 <= x <= 64]
        mbs = [1, 2, 4]
        min_model_parallel = 128
        max_model_parallel = 256
        gbs = 1920
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _tp_pp_mbs_grid_bert_80gb(model_size_in_b: float, valid_pp: List[int], model_measure: str) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for BERT and 80GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 3, 4, 6, 8]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1024
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if model_size_in_b <= 1.0:
        tp = [1, 2]
        gbs = 256
    elif model_size_in_b <= 4.0:
        tp = [1, 2, 4]
        gbs = 1024
    elif model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        min_model_parallel = 2
        gbs = 2048
    elif model_size_in_b <= 13.0:
        tp = [2, 4, 8]
        mbs = [1, 2, 3, 4, 6]
        min_model_parallel = 2
        gbs = 2048
    elif model_size_in_b <= 25.0:
        tp = [4, 8]
        mbs = [1, 2, 3, 4]
        min_model_parallel = 4
        gbs = 2048
    elif model_size_in_b <= 46.5:
        tp = [4, 8]
        pp = [1, 2, 4]
        mbs = [1, 2, 3, 4]
        min_model_parallel = 4
        max_model_parallel = 16
        gbs = 2048
    elif model_size_in_b <= 87.5:
        tp = [4, 8]
        pp = [2, 4, 6, 8]
        mbs = [1, 2, 3, 4]
        min_model_parallel = 8
        max_model_parallel = 32
        gbs = 2048
    elif model_size_in_b <= 165.5:
        tp = [4, 8]
        pp = [4, 6, 8, 16]
        mbs = [2, 4, 6, 8]
        min_model_parallel = 16
        max_model_parallel = 128
        gbs = 2048
    elif model_size_in_b <= 250.5:
        tp = [8]
        pp = [4, 8, 16, 32]
        mbs = [1, 2, 3, 4]
        min_model_parallel = 32
        max_model_parallel = 256
        gbs = 2048
    else:
        raise ValueError("No BERT model larger than 250B parameters is supported.")
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _tp_pp_mbs_grid_bert_40gb(model_size_in_b: float, valid_pp: List[int], model_measure: str) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for BERT and 40GB GPUs.
    :param float model_size_in_b: number of parameters in the model.
    :param List[int] valid_pp: list of valid Pipeline Parallelism (PP) values for this config.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
    """
    pp = [1]
    cp = [None]
    ep = [None]
    mbs = [1, 2, 4, 6, 8]
    min_model_parallel = 1
    max_model_parallel = 8
    gbs = 1024
    model_size_in_b = model_size_in_b / 1000 if model_measure == "M" else model_size_in_b
    if model_size_in_b <= 1.0:
        tp = [1, 2, 4]
        gbs = 256
    elif model_size_in_b <= 4.0:
        tp = [1, 2, 4, 8]
        gbs = 1024
    elif model_size_in_b <= 8.0:
        tp = [2, 4, 8]
        mbs = [1, 2, 4]
        gbs = 2048
    elif model_size_in_b <= 13.0:
        tp = [2, 4, 8]
        mbs = [1, 2, 4]
        gbs = 2048
    elif model_size_in_b <= 25.0:
        tp = [2, 4, 8]
        pp = [1, 2]
        mbs = [1, 2, 4]
        min_model_parallel = 2
        max_model_parallel = 16
        gbs = 2048
    elif model_size_in_b <= 46.5:
        tp = [4, 8]
        pp = [1, 2, 4, 8]
        mbs = [1, 2, 3]
        min_model_parallel = 8
        max_model_parallel = 32
        gbs = 2048
    elif model_size_in_b <= 87.5:
        tp = [4, 8]
        pp = [2, 4, 6, 8]
        mbs = [1, 2, 3]
        min_model_parallel = 16
        max_model_parallel = 64
        gbs = 2048
    elif model_size_in_b <= 165.5:
        tp = [8]
        pp = [4, 6, 8, 16]
        mbs = [1, 2]
        min_model_parallel = 32
        max_model_parallel = 256
        gbs = 2048
    elif model_size_in_b <= 250.5:
        tp = [8]
        pp = [8, 16, 32]
        mbs = [1, 2]
        min_model_parallel = 64
        max_model_parallel = 512
        gbs = 2048
    else:
        raise ValueError("No BERT model larger than 250B parameters is supported.")
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs


def _calculate_tp_pp_mbs_grid(
    model_size_in_b: float,
    num_layers: int,
    model_name: str,
    seq_length: int,
    train_cfg: dict,
) -> Tuple[int, int, int]:
    """
    Selects grid search space for TP, PP, MBS parameters for any model, and calls the necessary
    heuristics function accordingly.
    :param float model_size_in_b: number of parameters in the model.
    :param int num_layers: number of layers in the model config.
    :param str model_name: name of the model to be used, such as gpt3, t5, mt5...
    :param omegaconf.dictconfig.DictConfig train_cfg: config of the model that will be launched.
    :returns: tuple (tp, pp, cp, ep, mbs)
        WHERE
        int tp is the Tensor Parallelism value to use for training.
        int pp is the Pipeline Parallelism value to use for training.
        int cp is the Context Parallelism value to use for training.
        int ep is the Expert Parallelism value to use for training.
        int mbs is the Micro Batch Size to use for training.
        int min_model_parallel is the minimum parallelism level needed.
    :raises NotImplementedError: if the model_name is not one of the supported models.
    """
    tp_sizes = train_cfg.get("tensor_parallel_sizes")
    pp_sizes = train_cfg.get("pipeline_parallel_sizes")
    cp_sizes = train_cfg.get("context_parallel_sizes", None)
    ep_sizes = train_cfg.get("expert_parallel_sizes", None)
    min_model_parallel_size = train_cfg.get("min_model_parallel_size")
    max_model_parallel_size = train_cfg.get("max_model_parallel_size")
    mbs_sizes = train_cfg.get("micro_batch_sizes")
    gbs_size = train_cfg.get("global_batch_size")
    gpu_memory_gb = train_cfg.get("gpu_memory_gb")
    model_measure = train_cfg.get("model_measure")

    multiplier = (
        1 if model_name in ["gpt3", "bert", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"] else 2
    )
    init_pp = [] if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"] else [1]
    valid_pp = init_pp + [
        multiplier * x for x in range(1, num_layers + 1) if num_layers % x == 0
    ]  # Only divisors of num_layers are possible.

    if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral", "mistral"]:
        if gpu_memory_gb == 80:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_gpt3_80gb(
                model_size_in_b=model_size_in_b,
                valid_pp=valid_pp,
                seq_length=seq_length,
                model_measure=model_measure,
            )
        elif gpu_memory_gb == 40:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_gpt3_40gb(
                model_size_in_b=model_size_in_b, valid_pp=valid_pp, model_measure=model_measure
            )
    elif model_name in ["t5", "mt5"]:
        if gpu_memory_gb == 80:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_t5_80gb(
                model_size_in_b=model_size_in_b, valid_pp=valid_pp, model_measure=model_measure
            )
        elif gpu_memory_gb == 40:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_t5_40gb(
                model_size_in_b=model_size_in_b, valid_pp=valid_pp, model_measure=model_measure
            )
    elif model_name == "bert":
        if gpu_memory_gb == 80:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_bert_80gb(
                model_size_in_b=model_size_in_b, valid_pp=valid_pp, model_measure=model_measure
            )
        elif gpu_memory_gb == 40:
            (
                tp,
                pp,
                cp,
                ep,
                mbs,
                min_model_parallel,
                max_model_parallel,
                gbs,
            ) = _tp_pp_mbs_grid_bert_40gb(
                model_size_in_b=model_size_in_b, valid_pp=valid_pp, model_measure=model_measure
            )
    else:
        raise NotImplementedError("Model name not implemented.")

    # Override the tp, pp, mbs search if indicated in the config params.
    if tp_sizes is not None and tp_sizes != "auto":
        tp = tp_sizes
    if pp_sizes is not None and pp_sizes != "auto":
        pp = pp_sizes
    if cp_sizes is not None and cp_sizes != "auto":
        cp = cp_sizes
    if ep_sizes is not None and ep_sizes != "auto":
        ep = ep_sizes
    if mbs_sizes is not None and mbs_sizes != "auto":
        mbs = mbs_sizes
    if gbs_size is not None and mbs_sizes != "auto":
        gbs = gbs_size
    if min_model_parallel_size is not None and min_model_parallel_size != "auto":
        min_model_parallel = min_model_parallel_size
    if max_model_parallel_size is not None and max_model_parallel_size != "auto":
        max_model_parallel = max_model_parallel_size
    return tp, pp, cp, ep, mbs, min_model_parallel, max_model_parallel, gbs
