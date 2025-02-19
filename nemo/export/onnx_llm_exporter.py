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

from pathlib import Path
from typing import Optional, Union, Generator, List
import warnings
from tqdm import tqdm

import numpy as np
import tensorrt as trt
import torch
import wrapt
from transformers import AutoModel, AutoTokenizer
import json
import modelopt.torch.quantization as mtq

from nemo.deploy import ITritonDeployable
from nemo.export.utils import get_example_inputs, get_model_device_type, is_nemo2_checkpoint, validate_fp8_network


@wrapt.decorator
def noop_decorator(func):
    """No op decorator"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    warnings.warn("PyTriton is not available.")
    use_pytriton = False

use_onnxruntime = True
try:
    import onnxruntime
except Exception:
    warnings.warn("onnxruntime is not available.")
    use_onnxruntime = False


# pylint: disable=line-too-long
class OnnxLLMExporter(ITritonDeployable):
    """
    Exports nemo checkpoints to TensorRT-LLM and run fast inference.

    Example:
        from nemo.export.onnx_llm_exporter import OnnxLLMExporter

        trt_llm_exporter = OnnxLLMExporter(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)

    """

    def __init__(
        self,
        model_dir: str,
        model_name_or_path: str = None,
        load_runtime: bool=True,
    ):
        """
        Args:
            model_dir (str): path for storing the ONNX model files.
        """
        self.model_dir = model_dir
        self.model_name_or_path = model_name_or_path
        self.model_path = str(Path(model_dir) / "model.onnx")
        self.model = None
        self.tokenizer = None
        self.model_input_names = None
        self.model_output_names = None
        self.onnx_runtime_session = None
        self.calibration_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quant_max_batch_size = None

        if self.model_name_or_path is not None:
            self._load_hf_model()

        if load_runtime:
            self._load_runtime()

    def _load_runtime(self):
        if use_onnxruntime:
            if Path(self.model_path).exists():
                self.onnx_runtime_session = onnxruntime.InferenceSession(self.model_path)
                self.model_input_names = [input.name for input in self.onnx_runtime_session.get_inputs()]
                self.model_output_names = [output.name for output in self.onnx_runtime_session.get_outputs()]
                self.tokenizer = AutoTokenizer.from_pretrained(Path(self.model_dir) / "tokenizer", trust_remote_code=True)

    def _load_hf_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)

    def export(
        self,
        input_names: list,
        output_names: list,
        model_name_or_path: str = None,
        example_inputs: dict = None,
        opset: int = 20,
        dynamic_axes_input: Optional[dict] = None,
        dynamic_axes_output: Optional[dict] = None,
        export_dtype: str = "fp32",
        model_dtype: Optional[Union[torch.dtype, str]] = None,
        verbose: bool = False,
    ):
        """Performs ONNX conversion from a PyTorch model."""

        reload_model = False
        if model_name_or_path is not None:
            if self.model_name_or_path is not None:
                warnings.warn("The model name or path was set before and it is being updated.")
            self.model_name_or_path = model_name_or_path
            reload_model = True

        assert self.model_name_or_path is not None, "Model name or path cannot be None"

        is_nemo_ckpt = False
        if Path(self.model_name_or_path).is_dir():
            if is_nemo2_checkpoint(self.model_name_or_path):
                is_nemo_ckpt = True

        if is_nemo_ckpt:
            self._export_nemo_to_onnx(
                input_names=input_names,
                example_inputs=example_inputs,
                output_names=output_names,
                opset=opset,
                dynamic_axes_input=dynamic_axes_input,
                dynamic_axes_output=dynamic_axes_output,
                export_dtype=export_dtype,
            )
        else:
            self._export_hf_to_onnx(
                input_names=input_names,
                example_inputs=example_inputs,
                output_names=output_names,
                opset=opset,
                dynamic_axes_input=dynamic_axes_input,
                dynamic_axes_output=dynamic_axes_output,
                export_dtype=export_dtype,
                model_dtype=model_dtype,
                trust_remote_code=True,
                reload_model=reload_model,
                verbose=verbose,
            )

        self._load_runtime()

    def _export_hf_to_onnx(
        self,
        input_names: list,
        output_names: list,
        example_inputs: dict = None,
        opset: int = 20,
        dynamic_axes_input: Optional[dict] = None,
        dynamic_axes_output: Optional[dict] = None,
        export_dtype: Union[torch.dtype, str] = "fp32",
        model_dtype: Optional[Union[torch.dtype, str]] = None,
        trust_remote_code: bool = True,
        reload_model: bool = False,
        verbose: bool = False,
    ):

        if reload_model:
            self._load_hf_model()

        if example_inputs is None:
            example_inputs = get_example_inputs(self.tokenizer)

        if "dimensions" in input_names:
            example_inputs["dimensions"] = torch.tensor([1] * example_inputs["input_ids"].shape[0])

        if isinstance(export_dtype, str):
            export_dtype = {"fp16": torch.float16, "fp32": torch.float32}[export_dtype]

        self.model.to(export_dtype)

        with torch.autocast(device_type=get_model_device_type(self.model), dtype=export_dtype):
            torch.onnx.export(
                model=self.model,
                args=(example_inputs,),
                f=self.model_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={**dynamic_axes_input, **dynamic_axes_output},
                verbose=verbose,
                opset_version=opset,
            )
        print(f"Successfully exported PyTorch model to " f"ONNX model ({self.model_path})")

        existing_directory_path = Path(self.model_dir) / "tokenizer"
        existing_directory_path.mkdir(exist_ok=True)
        self.tokenizer.save_pretrained(existing_directory_path)

    def _export_nemo_to_onnx(
        self,
        input_names: list,
        example_inputs: dict,
        output_names: list,
        opset: int = 20,
        dynamic_axes_input: Optional[dict] = None,
        dynamic_axes_output: Optional[dict] = None,
        export_dtype: str = "fp32",
        verbose: bool = False,
    ):
        raise NotImplementedError("This function will be implemented later.")

    def export_onnx_to_trt(
        self,
        trt_model_path,
        profiles=None,
        override_layernorm_precision_to_fp32=False,
        override_layers_to_fp32=None,
        trt_dtype="fp16",
        profiling_verbosity="layer_names_only",
    ) -> None:
        """Performs TensorRT conversion from an ONNX model.

        Raises:
            SerializationError: TensorRT engine must serialize properly.
        """
        print(f"Building TRT engine from ONNX model ({self.model_path})")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, trt_logger)

        # we use parse_from_file() instead of parse() because it can be used for both single
        # file models as well as externally stored models (required when model >2GiB)
        if not parser.parse_from_file(self.model_path):
            print("ONNX model could not be parsed")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

        if profiles:
            for profile in profiles:
                optimization_profile = builder.create_optimization_profile()

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    optimization_profile.set_shape(
                        in_tensor.name,
                        min=profile[in_tensor.name][0],
                        opt=profile[in_tensor.name][1],
                        max=profile[in_tensor.name][2],
                    )

                config.add_optimization_profile(optimization_profile)

        if trt_dtype == "fp16":
            print("Setting Build Flag FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        elif trt_dtype == "fp8":
            # With FP8 export we want to also enable FP16 layers as a fallback instead of FP32
            print("Setting Build Flag FP8 and FP16")
            config.set_flag(trt.BuilderFlag.FP8)
            config.set_flag(trt.BuilderFlag.FP16)
            validate_fp8_network(network)

        # patch network
        if override_layernorm_precision_to_fp32:
            print("Overriding TensorRT network LayerNorm precision to float32.")
            self._override_layernorm_precision_to_fp32(network)

        if override_layers_to_fp32:
            print("Overriding some layers to float32.")
            self._override_layers_to_fp32(network, override_layers_to_fp32)

        try:
            config.profiling_verbosity = {
                "detailed": trt.ProfilingVerbosity.DETAILED,
                "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                "none": trt.ProfilingVerbosity.NONE,
            }[profiling_verbosity]
        except KeyError:
            error_msg = f"Unknown profiling verbosity value "
            raise ValueError(error_msg)
        print(f"Setting Profiling Verbosity to {config.profiling_verbosity}")

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            raise Exception("Failed to serialize the TensorRT Engine. Please check the " "TensorRT logs for details")

        trt_model_path.write_bytes(engine_string)
        print(f"Successfully exported ONNX model ({self.model_path}) " f"to TRT engine ({trt_model_path})")

    def _override_layer_precision_to_fp32(self, layer: trt.ILayer) -> None:
        """Set TensorRT layer precision and output type to FP32.

        Args:
            layer: tensorrt.ILayer.
        """
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)

    def _override_layers_to_fp32(self, network: trt.INetworkDefinition, fp32_layer_patterns: list[str]) -> None:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            if any(layer_name.startswith(pattern) for pattern in fp32_layer_patterns) and layer.precision in {
                trt.float32,
                trt.float16,
            }:
                if layer.type in {trt.LayerType.CAST}:
                    print(f"Skipping overriding {layer.type} layer {i} {layer_name} dtype")
                    continue
                print(layer_name)
                if any(
                    layer.get_input(input_idx).dtype in {trt.float32, trt.float16}
                    for input_idx in range(layer.num_inputs)
                ):
                    # Note: Assigning to layer.precision (even the same value) sets precision_is_set=True,
                    # which prevents TensorRT from changing this layer's precision
                    layer.precision = trt.float32
                    print(f"Setting layer {i} {layer_name} (type: {layer.type}) precision to FP32")
                for j in range(layer.num_outputs):
                    if layer.get_output_type(j) in {trt.float32, trt.float16}:
                        layer.set_output_type(j, trt.float32)
                        print(f"Setting layer {i} {layer_name} (type: {layer.type}) output type {j} to FP32")

    def _override_layernorm_precision_to_fp32(self, network: trt.INetworkDefinition) -> None:
        """Set the precision of LayerNorm subgraphs to FP32 to preserve accuracy.

        - https://nvbugs/4478448 (Mistral)
        - https://nvbugs/3802112 (T5)

        Args:
            network: tensorrt.INetworkDefinition
        """
        # Logic originally from OSS T5 HF export script:
        # https://gitlab-master.nvidia.com/TensorRT/Public/oss/-/blob/77495ec/demo/HuggingFace/T5/export.py
        pow_ops = {}
        for layer_index, layer in enumerate(network):
            if layer.type == trt.LayerType.IDENTITY:
                all_fp32 = all(
                    [
                        layer.output_type_is_set(o) and layer.get_output_type(o) == trt.float32
                        for o in range(layer.num_outputs)
                    ]
                )
                if all_fp32:
                    if layer.get_input(0).dtype == trt.float32:
                        layer.precision = trt.float32

            if layer.type == trt.LayerType.ELEMENTWISE:
                layer.__class__ = getattr(trt, "IElementWiseLayer")
                if layer.op == trt.ElementWiseOperation.POW:
                    pow_ops[layer] = layer_index
                    self._override_layer_precision_to_fp32(layer)

        for _, index in pow_ops.items():
            # Iterate from few layers before pow to include residual add and cast op.
            # Iterate till 10 layers after pow op to include all
            # operations included in layer norm.
            START_OFFSET = 4
            END_OFFSET = 12
            for i in range(index - START_OFFSET, index + END_OFFSET):
                layer = network.get_layer(i)
                if layer.type == trt.LayerType.REDUCE:
                    self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.SUM:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.UNARY:
                    layer.__class__ = getattr(trt, "IUnaryLayer")
                    if layer.op == trt.UnaryOperation.SQRT:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.DIV:
                        self._override_layer_precision_to_fp32(layer)

                if layer.type == trt.LayerType.ELEMENTWISE:
                    layer.__class__ = getattr(trt, "IElementWiseLayer")
                    if layer.op == trt.ElementWiseOperation.PROD:
                        self._override_layer_precision_to_fp32(layer)

    def ptq(
        self,
        calibration_data: str,
        quantization_type="fp8",
        max_batch_size=32,
    ) -> None:
        """Runs a calibration loop on the model using a calibration dataset."""

        if Path(self.model_name_or_path).is_dir():
            if is_nemo2_checkpoint(self.model_name_or_path):
                raise NotImplementedError("NeMo 2.0 model is not currently supported.")

        self.quant_max_batch_size = max_batch_size

        quant_cfg_choices = {
            "int8": mtq.INT8_DEFAULT_CFG,
            "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
            "fp8": mtq.FP8_DEFAULT_CFG,
            "int4_awq": mtq.INT4_AWQ_CFG,
            "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        }
        quant_cfg = quant_cfg_choices[quantization_type]

        # Enable FP8 kv cache to save memory footprint
        quant_cfg["quant_cfg"]["*output_quantizer"] = {
            "num_bits": (
                8
                if quantization_type == "int8_sq" or quantization_type == "int8"
                else (4, 3)
            ),
            "axis": None,
            "enable": True,
        }

        with open(calibration_data, mode="r") as _file:
            self.calibration_data = json.load(_file)

        self.model.to(self.device, dtype=torch.float16)
        print("Starting quantization...")

        mtq.quantize(self.model, quant_cfg, forward_loop=self._calibrate_loop)
        self.model.to("cpu").to(torch.float32)
        print("Done ...")

        return self.model

    def _calibrate_loop(self, model) -> None:

        def chunk(iterable: List, n: int) -> Generator:
            for i in range(0, len(iterable), n):
                yield iterable[i : i + n]  # noqa: E203

        pbar = tqdm(total=len(self.calibration_data), desc="Running calibration loop")

        for batch in chunk(self.calibration_data, self.quant_max_batch_size):
            inputs = [f"question:{query} \n \n passage:{passage}" for query, passage in batch]
            batch = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)
            torch.cuda.empty_cache()
            del batch
            pbar.update(self.quant_max_batch_size)
            pbar.refresh()

    def forward(self, prompt):
        if self.onnx_runtime_session is None:
            warnings.warn("ONNX Runtime is not available.")
            return None
        else:
            tokenized = self.tokenizer(prompt)
            input_data = {nn: tokenized[nn] for nn in self.model_input_names}

            output = self.onnx_runtime_session.run(self.model_output_names, input_data)
            return output[0][0]

    @property
    def get_triton_input(self):
        """Get triton input"""
        raise NotImplementedError("This function will be implemented later.")

    @property
    def get_triton_output(self):
        raise NotImplementedError("This function will be implemented later.")

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        raise NotImplementedError("This function will be implemented later.")
