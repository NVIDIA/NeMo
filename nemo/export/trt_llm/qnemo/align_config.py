import copy
from typing import Any, Dict


def align_config(config_trtllm_build: Dict[str, Any]) -> Dict[str, Any]:
    """Function to align config produced by trtllm-build API for consistency
    with how ModelConfig from tensorrt_llm.runtime is used in the project.
    """
    config = {}

    config_trtllm_build = copy.deepcopy(config_trtllm_build)

    # Builder config
    config["builder_config"] = {}
    config["builder_config"]["name"] = "NeMo"
    config["builder_config"].update(config_trtllm_build["build_config"])
    config["builder_config"].update(config_trtllm_build["pretrained_config"])

    # Plugin config
    config["plugin_config"] = config["builder_config"].pop("plugin_config")

    # Parallelism config
    config["builder_config"]["world_size"] = config["builder_config"]["mapping"]["world_size"]
    config["builder_config"]["tensor_parallel"] = config["builder_config"]["mapping"]["tp_size"]
    config["builder_config"]["pipeline_parallel"] = config["builder_config"]["mapping"]["pp_size"]

    # Other parameters
    config["builder_config"]["num_heads"] = config_trtllm_build["pretrained_config"]["num_attention_heads"]
    config["builder_config"]["num_layers"] = config_trtllm_build["pretrained_config"]["num_hidden_layers"]
    config["builder_config"]["add_bos"] = False
    config["builder_config"]["precision"] = config["builder_config"]["dtype"]
    return config
