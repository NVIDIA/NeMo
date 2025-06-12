# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List

import nemo_run as run
from lightning.pytorch.callbacks.callback import Callback

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model import GPTModel
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def hf_tokenizer(model_name: str) -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    log_msg = [
        f"`AutoTokenizer` first searches for tokenizer files locally stored in {DEFAULT_NEMO_HOME}.",
        "(from env var `NEMO_HOME`- can be changed using '-nh/--nemo_home' CLI arg).",
        "If files are missing locally, `AutoTokenizer` will try downloading from HuggingFace. In this case-",
        "make sure env vars 'TRANSFORMERS_OFFLINE':'0' and 'HF_TOKEN':'<token_value>' are set in your sbatch script.",
        "Both of these will be set automatically if you provide '-hf/--hf_token' CLI arg.",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


def import_ckpt_experiment(executor: run.SlurmExecutor, model: run.Config[GPTModel], source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor, "import_ckpt_exp"


def isfile_train_pack_metadata(hf_model_uri: str, data_config: run.Config[SquadDataModule]) -> bool:
    """
    This method is used for fine-tuning. It checks if packed train data for a partiular
    sequence length exists locally. This is needed to set data flag (force_redownload=True)
    which avoids experiment crash in case files are missing.
    """
    datasets_dir = os.getenv("NEMO_DATASETS_CACHE", os.path.join(DEFAULT_NEMO_HOME, "datasets"))
    model_dir = hf_model_uri.replace("/", "--")
    metadata_filename = f"{data_config.seq_length}_metadata.jsonl"

    train_pack_metadata_filepath = os.path.join(datasets_dir, "squad", "packed", model_dir, metadata_filename)

    return os.path.exists(train_pack_metadata_filepath) and os.path.isfile(train_pack_metadata_filepath)


def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None
