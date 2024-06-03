import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from safetensors import safe_open
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from transformers import PreTrainedTokenizer

from nemo.export.tarutils import unpack_tarball
from nemo.export.trt_llm.nemo.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.export.trt_llm.qnemo.tokenizer_utils import get_nmt_tokenizer
from nemo.export.trt_llm.qnemo.checkpoint_utils import CONFIG_NAME, WEIGHTS_NAME


def load_weights(weights_file: str) -> Dict[str, torch.Tensor]:
    weights = {}
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    return weights


def qnemo_to_trtllm_config(
    in_file: str,
    decoder_type: str,
    nemo_export_dir: Union[str, Path],
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
) -> Tuple[List[Dict[str, torch.Tensor]], List[PretrainedConfig], Union[PreTrainedTokenizer, SentencePieceTokenizer]]:
    """Prepare weights and model config from qnemo file for TensorRT-LLM engine build."""
    print(
        "Note that setting tensor_parallel_size and pipeline_parallel_size parameters for quantized"
        " models is possible only on the checkpoint export step via nemo.export.quantize module."
        " These parameters are read out from config.json in the case of TensorRT-LLM checkpoint."
    )

    if os.path.isfile(in_file):  # qnemo is a tarball
        nemo_export_dir = str(nemo_export_dir)
        unpack_tarball(in_file, nemo_export_dir)
        in_file = nemo_export_dir

    model_config = PretrainedConfig.from_json_file(os.path.join(in_file, CONFIG_NAME))
    # TODO: Llama naming inconsistency: LLaMAForCausalLM vs LLamaForCausalLM
    if model_config.architecture == "LlamaForCausalLM":
        assert decoder_type == "llama", f"decoder_type mismatch: {decoder_type}"
        model_config.architecture = "LLaMAForCausalLM"

    world_size = model_config.mapping.world_size

    model_configs = [model_config]
    for i in range(1, world_size):
        model_config_i = PretrainedConfig.from_dict(model_config.to_dict())  # copy
        model_config_i.set_rank(i)
        model_configs.append(model_config_i)

    weights = [load_weights(os.path.join(in_file, WEIGHTS_NAME.format(i))) for i in range(world_size)]
    tokenizer = get_nmt_tokenizer(in_file)
    return weights, model_configs, tokenizer
