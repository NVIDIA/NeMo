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

import argparse
import shutil
import os

import torch
import torch.distributed as dist

from nemo import lightning as nl
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint

    QUANT_CFG_CHOICES = {
        "int8": mtq.INT8_DEFAULT_CFG,
        "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
    }

    HAVE_MODELOPT = True

except (ImportError, ModuleNotFoundError) as e:
    HAVE_MODELOPT = False
    HAVE_MODELOPT_ERROR = e


SUPPORTED_DTYPE = [16, "16", "bf16"]  # Default precision for non-quantized layers

# TODO: delete
class config_dict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _dict_to_config(config):
    if isinstance(config, dict):
        return config_dict(config)
    return config


#### nemo.export.quantize.quantizer.Quantizer class for NeMo 2
class Quantizer:
    """Post-training quantization (PTQ) and TRT-LLM export of Nemo checkpoints.

    PTQ converts selected model layers to low-precision format (e.g., INT4, FP8) for efficient serving.
    The process consist of several steps:

        1. Loading a Nemo model from disk using appropriate parallelism strategy
        2. Calibrating the model to obtain appropriate algorithm-specific scaling factors
        3. Producing output directory or .qnemo tarball with model config (json),
           quantized weights (safetensors) and tokenizer config (yaml).

    The output directory (or .qnemo file) produced is intended to be consumed by TensorRT-LLM toolbox
    for efficient inference. This can be achieved using Nemo inference containers.

    Currently supported and tested model family is Llama2. Model type needs to be specified in
    the quantization command with decoder_type parameter on exporting (see below). Quantizing other
    model families is experimental and might not be fully supported.

    Available quantization methods are listed in `QUANT_CFG_CHOICES` dictionary above.
    Please consult Model Optimizer documentation https://nvidia.github.io/TensorRT-Model-Optimizer/ for details.
    You can also inspect different choices in examples/nlp/language_modeling/conf/megatron_gpt_ptq.yaml
    for quantization algorithms and calibration data as well as recommended settings.

    Quantization algorithm can also be conveniently set to 'null' to perform only weights export step
    for TensorRT-LLM deployment. This is useful to getting baseline results for a full-precision model.
    """

    def __init__(self, quantization_config, export_config):
        """Initialize Quantizer with quantization and export configurations.

        Expected keys in `quantization_config`:
            - algorithm: str
            - awq_block_size: int (only for awq algorithms)
            - sq_alpha: float (only for smooth quant algorithms)
            - enable_kv_cache: bool (default: None i.e. auto-detect based on algorithm and decoder_type)

        Expected keys in `export_config`:
            - dtype: str/int
            - decoder_type: str
            - inference_tensor_parallel: int
            - inference_pipeline_parallel: int
            - path: str
        """
        if not HAVE_MODELOPT:
            raise RuntimeError("nvidia-modelopt is needed to use Quantizer") from HAVE_MODELOPT_ERROR
        if not torch.cuda.is_available():
            raise EnvironmentError("GPU is required for the quantization.")

        quantization_config = _dict_to_config(quantization_config)
        export_config = _dict_to_config(export_config)

        self.quantization_config = quantization_config
        self.export_config = export_config

        # Quantization sanity checks
        assert (
            quantization_config.algorithm is None or quantization_config.algorithm in QUANT_CFG_CHOICES
        ), f"Unsupported quantization algorithm: {quantization_config.algorithm}"
        if quantization_config.algorithm is not None:
            quant_cfg = QUANT_CFG_CHOICES[quantization_config.algorithm]

            if "awq" in quantization_config.algorithm:
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                weight_quantizer["block_sizes"][-1] = quantization_config.awq_block_size

            # Always turn on FP8 kv cache to save memory footprint.
            # For int8_sq, we use int8 kv cache.
            # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for Nemotron.
            enable_quant_kv_cache = quantization_config.get("enable_kv_cache", None)
            if enable_quant_kv_cache is None:
                enable_quant_kv_cache = (
                    "int8" not in quantization_config.algorithm and quantization_config.decoder_type != "gptnext"
                )
            logging.info(f'{"Enabled" if enable_quant_kv_cache else "Disabled"} KV cache quantization')
            quant_cfg["quant_cfg"]["*output_quantizer"] = {
                "num_bits": 8 if quantization_config.algorithm == "int8_sq" else (4, 3),
                "axis": None,
                "enable": enable_quant_kv_cache,
            }
            if quantization_config.algorithm == "int8_sq":
                logging.info(f"Using int8_sq alpha = {quantization_config.sq_alpha}")
                quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": quantization_config.sq_alpha}

            self.quant_cfg = quant_cfg
        else:
            self.quant_cfg = None

        # Export sanity checks
        if export_config is not None:
            assert export_config.dtype in SUPPORTED_DTYPE, f"Unsupported export dtype: {export_config.dtype}"

        self.nemo_checkpoint_path = None


    def load_quantizable_model(self, nemo_checkpoint_path: str, tensor_parallelism_size: int = 1):
        from nemo.collections.nlp.models.language_modeling.megatron.gpt_layer_modelopt_spec import get_gpt_layer_modelopt_spec

        self.nemo_checkpoint_path = nemo_checkpoint_path

        trainer = nl.Trainer(
            devices=tensor_parallelism_size,
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=tensor_parallelism_size,
                pipeline_model_parallel_size=1,
                ),
            plugins=nl.MegatronMixedPrecision(precision='16-mixed'),
        )
        fabric = trainer.to_fabric()
        trainer.strategy.setup_environment()

        model = nl.io.load_context(nemo_checkpoint_path).model
        model.config.transformer_layer_spec = get_gpt_layer_modelopt_spec()
        model.config = self.modify_model_config(model.config)

        # TODO: [0] works only for PP=1
        model = fabric.load_model(nemo_checkpoint_path, model=model)[0]
        model.freeze()
        return model


    # TODO: what happens with NeMo 2?
    # @staticmethod
    # def _setup(model):
    #     """Setup model for quantization."""
    #     try:
    #         model.model.module.language_model.encoder.activations_checkpoint_method = None
    #     except AttributeError:
    #         pass

    #     if not parallel_state.is_initialized():

    #         def dummy():
    #             return

    #         if model.trainer.strategy.launcher is not None:
    #             model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
    #         model.trainer.strategy.setup_environment()

    @staticmethod
    def create_argparser():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="NeMo PTQ argument parser",
        )
        parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source NeMo 2.0 checkpoint")
        parser.add_argument("--decoder_type", type=str, help="Decoder type for TensorRT-Model-Optimizer")
        parser.add_argument(
            "-tps",
            "--tensor_parallelism_size",
            type=int,
            default=1
        )
        parser.add_argument(
            '-out',
            '--output_path',
            type=str,
            help='Path for the exported engine'
        )
        parser.add_argument(
            '--quant_algo',
            type=str,
            default="no_quant",
            choices=["no_quant", "int8", "int8_sq", "fp8", "int4_awq", "w4a8_awq", "int4"],
            help='TensorRT-Model-Optimizer quantization algorithm'
        )
        parser.add_argument(
            '-awq_bs',
            '--awq_block_size',
            type=int,
            default=128,
            help='Block size for AWQ quantization algorithms'
        )
        parser.add_argument(
            '--sq_alpha',
            type=float,
            default=1.0,
            help='Smooth-Quant alpha parameter'
        )
        
        return parser

    @staticmethod
    def postprocess_argparse(args):
        args.quantization_config = {
            "algorithm": args.quant_algo,
            "awq_block_size": args.awq_block_size,
            "sq_alpha": args.sq_alpha,
            "enable_kv_cache": None,
        }
        args.export_config = {
            "path": args.output_path,
            "decoder_type": args.decoder_type,
            "inference_tensor_parallel": args.tensor_parallelism_size,
            "inference_pipeline_parallel": 1,
            "dtype": "bf16",
        }
        return args

    @staticmethod
    def modify_model_config(model_cfg):
        """Modify model config for quantization."""

        # TODO: re-think
        # with open_dict(model_cfg):
        if model_cfg.sequence_parallel:
            logging.warning("Disabling sequence parallelism for quantization...")
            model_cfg.sequence_parallel = False
        # Only custom ModelOpt spec is supported for Quantization: this custom spec is largely based on local Megatron-LM
        # layer definitions to avoid Transformer Engine implementations that are currently not supported.
        # This layer spec also requires RoPE fusion to be disabled for tensor view operations in attention
        # layer implementation from megatron/core/transformer/dot_product_attention.py to be functional.
        model_cfg.name = "modelopt"
        model_cfg.apply_rope_fusion = False

        return model_cfg

    # TODO: Add support for NeMo 2
    # @staticmethod
    # def _sample_output(model: MegatronGPTModel):
    #     """Generate sample output for a model instance."""
    #     logging.info("Generating sample output for the model...")

    #     response = model.generate(
    #         inputs=[
    #             "Born in north-east France, Soyer trained as a",
    #             "Born in California, Soyer trained as a",
    #         ],
    #         length_params={
    #             "max_length": 100,
    #             "min_length": 100,
    #         },
    #     )

    #     logging.info(f'Example NeMo output before export: {response["sentences"]}"')

    def quantize(self, model, forward_loop):
        """Quantize the model and calibrate using given forward loop."""
        assert self.quant_cfg is not None, "Quantization algorithm is not set"

        logging.info(f"Quantizing model to {self.quantization_config.algorithm}...")
        # self._setup(model)

        model = mtq.quantize(model, self.quant_cfg, forward_loop)

        if self.quantization_config.decoder_type == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if self.quantization_config.algorithm == "fp8":
                maxbound = 448
            elif self.quantization_config.algorithm == "int8_sq":
                maxbound = 127
            model = mtq.postprocess_amax(
                model, "*input_quantizer", lambda amax: torch.clamp(amax, min=0.01 * maxbound)
            )

        if dist.get_rank() == 0:
            mtq.print_quant_summary(model)

        return model

    def export(self, model, nemo_checkpoint_path = None):
        """Export model to '.qnemo' format for TensorRT-LLM engine build."""
        assert self.export_config is not None, "Export config is not set"
        torch_dtype = torch_dtype_from_precision(self.export_config.dtype)

        # TODO: add with generate
        # if self.export_config.get("sample_output", True):
        #     self._sample_output(model)


        # TODO: Support compressing to .qnemo
        
        # TODO: SUPPORT NeMo 2:
        # if model.cfg.megatron_amp_O2:
        #     model.model = unwrap_model(model.model, Float16Module)

        # with export_handler as export_dir:
        mtq.print_quant_summary(model)
        export_dir = self.export_config.path
        export_tensorrt_llm_checkpoint(
            model=model,
            decoder_type=self.export_config.decoder_type,
            dtype=torch_dtype,
            export_dir=export_dir,
            inference_tensor_parallel=self.export_config.inference_tensor_parallel,
            inference_pipeline_parallel=self.export_config.inference_pipeline_parallel,

            # TODO: What happens in NeMo 2?
            # use_nfs_workspace=model.trainer.num_nodes > 1,
        )
        dist.barrier()  # Wait until all ranks complete export_model_config step
        logging.info(
            f"Exporting quantized weights, model artifacts, and tokenizer config to {self.export_config.path}..."
        )

        
        if dist.get_rank() == 0:
            self.nemo_checkpoint_path = nemo_checkpoint_path or self.nemo_checkpoint_path

            if self.nemo_checkpoint_path is not None:
                tokenizer_src = os.path.join(self.nemo_checkpoint_path, 'nemo_tokenizer')
                tokenizer_dst = os.path.join(self.export_config.path, 'tokenizer')

                if os.path.exists(tokenizer_src):
                    shutil.copytree(tokenizer_src, tokenizer_dst)
                else:
                    print("Could not find copy tokenizer from NeMo checkpoint")

            # TODO Support for NeMo 2?
            # save_artifacts(model, export_dir)



def get_calib_data_iter(data="cnn_dailymail", batch_size=64, calib_size=512, max_sequence_length=512):
    from datasets import load_dataset
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    else:
        # Assume a local JSON dataset with a column named "text"
        dataset = load_dataset("json", data_files=data, split="train")
        text_column = "text"
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch
