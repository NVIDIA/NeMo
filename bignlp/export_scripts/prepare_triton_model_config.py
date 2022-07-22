#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import argparse
import configparser
import logging
import pathlib
import sys

import google.protobuf.json_format
import google.protobuf.text_format
import tritonclient.grpc

LOGGER = logging.getLogger(__name__)


def _get_model_parameters(config_ini):
    excluded_section_names = ["ft_instance_hyperparameter", "structure"]
    sections_names_with_model_parameters = [s for s in config_ini.sections() if s not in excluded_section_names]

    if not sections_names_with_model_parameters:
        LOGGER.error(
            "Could not find section with model parameters in model config.ini while it is required to fill templates"
        )
        sys.exit(-1)

    def _get_model_name(section_name_):
        model_name = config_ini.get(section_name_, "model_name", fallback=None)
        if model_name is None:
            model_name = config_ini.get(section_name_, "_name_or_path", fallback="unknown")
        return model_name

    params_from_model_config = {
        section_name: {
            "model_type": config_ini.get(section_name, "model_type", fallback="GPT"),
            "tensor_para_size": config_ini.getint(section_name, "tensor_para_size"),
        }
        for section_name in sections_names_with_model_parameters
    }

    # ensure that for all models it is obtained same parameters
    parameters_from_all_sections = list(set(map(lambda x: tuple(x.items()), params_from_model_config.values())))[0]
    if len(parameters_from_all_sections) != len(list(params_from_model_config.values())[0]):
        LOGGER.error(
            "Found no consistency between model parameters: %s (%d != %d)",
            params_from_model_config,
            len(parameters_from_all_sections),
            len(list(params_from_model_config.values())[0]),
        )
        sys.exit(-1)

    params_from_model_config = list(params_from_model_config.values())[0]
    return params_from_model_config


def _update_template(
    config, model_name, default_model_filename, max_batch_size, parameters, just_update_parameters: bool = True
):
    config["name"] = model_name
    config["default_model_filename"] = default_model_filename
    config["max_batch_size"] = max_batch_size

    parameters = {k: {"string_value": str(v)} for k, v in parameters.items()}
    replace = not just_update_parameters
    if replace:
        config["parameters"] = parameters
    else:
        # overwrite
        config["parameters"] = {**config["parameters"], **parameters}
    return config


def main():

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate Triton model config file")
    parser.add_argument("--model-train-name", help="Name of trained model", required=True)
    parser.add_argument("--template-path", help="Path to template of Triton model config file", required=True)
    parser.add_argument("--config-path", help="Path to output Triton model config file", required=True)
    parser.add_argument("--ft-checkpoint", help="Path to FasterTransformer checkpoint", required=True)
    parser.add_argument("--max-batch-size", type=int, help="Max batch size of Triton batcher", required=True)
    parser.add_argument("--pipeline-model-parallel-size", type=int, help="Pipeline model parallel size", required=True)
    parser.add_argument(
        "--data-type", choices=["fp32", "fp16", "bf16"], help="Data type of weights in runtime", required=True
    )
    parser.add_argument("--int8-mode", action="store_true", help="Enable int8 mode in FasterTransformer Triton backend")
    parser.add_argument(
        "--enable-custom-all-reduce",
        action="store_true",
        help="Enable custom all reduce ops in FasterTransformer Triton backend",
    )
    args = parser.parse_args()

    ft_checkpoint_path = pathlib.Path(args.ft_checkpoint)
    config_ini_path = ft_checkpoint_path / "config.ini"
    config_ini = configparser.ConfigParser()
    with config_ini_path.open("r") as config_file:
        config_ini.read_file(config_file)

    # parse template
    template_path = pathlib.Path(args.template_path)
    template_payload = template_path.read_text()
    model_config_proto = google.protobuf.text_format.Parse(
        template_payload, tritonclient.grpc.model_config_pb2.ModelConfig()
    )
    triton_model_config_template = google.protobuf.json_format.MessageToDict(
        model_config_proto, preserving_proto_field_name=True
    )

    # update template
    params_from_model_config = _get_model_parameters(config_ini)
    parameters = {
        **{
            "data_type": args.data_type.lower(),
            "pipeline_para_size": args.pipeline_model_parallel_size,
            "model_checkpoint_path": ft_checkpoint_path.as_posix(),
            "int8_mode": int(args.int8_mode),
            "enable_custom_all_reduce": int(args.enable_custom_all_reduce),
        },
        **params_from_model_config,
    }
    model_name = args.model_train_name
    updated_triton_model_config = _update_template(
        triton_model_config_template, model_name, ft_checkpoint_path.name, args.max_batch_size, parameters
    )

    # store template
    updated_triton_model_config = google.protobuf.json_format.ParseDict(
        updated_triton_model_config, tritonclient.grpc.model_config_pb2.ModelConfig()
    )
    updated_triton_model_config_payload = google.protobuf.text_format.MessageToBytes(updated_triton_model_config)

    config_path = pathlib.Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("wb") as config_file:
        config_file.write(updated_triton_model_config_payload)

    LOGGER.info("Config file successfully generated and written to: %s", config_path)


if __name__ == "__main__":
    main()
