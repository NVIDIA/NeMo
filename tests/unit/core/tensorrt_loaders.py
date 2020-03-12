# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import time
import warnings
from collections import OrderedDict

import numpy as np
import onnx
import tensorrt as trt

from .tensorrt_format import FormatManager
from .tensorrt_runner import (
    DEFAULT_SHAPE_VALUE,
    TRT_LOGGER,
    TensorRTRunnerV2,
    default_value,
    find_in_dict,
    get_input_metadata_from_profile,
    is_dimension_dynamic,
    is_shape_dynamic,
    is_valid_shape_override,
    send_on_queue,
    write_timestamped,
)
from nemo import logging, logging_mode


def set_onnx_logging_level(sev):
    if sev >= logging.INFO:
        warnings.filterwarnings("ignore")


class BaseDataLoader(object):
    """
    Responsible for fetching or generting input data for runners.
    """

    def __call__(self, index, input_metadata, input_example=None):
        """
        Fetches or generates inputs.

        Args:
            index (int): The index of inputs to fetch. For any given index, the inputs should always be the same.
            input_metadata (OrderedDict[str, Tuple[np.dtype, Tuple[int]]]): Mapping of input names to their data types and shapes.

        Returns:
            OrderedDict[str, np.ndarray]: Mapping of input names to numpy buffers containing data.
        """
        raise NotImplementedError("BaseDataLoader is an abstract class")


class DefaultDataLoader(BaseDataLoader):
    def __init__(
        self,
        seed=None,
        default_shape_value=None,
        default_shapes=None,
        int_min=None,
        int_max=None,
        float_min=None,
        float_max=None,
    ):
        """
        Optional Args:
            seed (int): The seed to use when generating random inputs.
            default_shape_value (int): The default value to use when a dimension is dynamic.
            default_shapes (Dict[str, Tuple[int]]): A mapping of input names to their corresponding shapes.
        """
        self.seed = default_value(seed, int(time.time()))
        self.default_shapes = default_value(default_shapes, {})
        self.default_shape_value = default_value(default_shape_value, DEFAULT_SHAPE_VALUE)
        self.int_min = default_value(int_min, 1)
        self.int_max = default_value(int_max, 25)
        self.float_min = default_value(float_min, -1.0)
        self.float_max = default_value(float_max, 1.0)

    def __call__(self, index, input_metadata, input_example=None):
        logging.debug("Updating seed to: {:}".format(self.seed + index))
        rng = np.random.RandomState(self.seed + index)

        buffers = OrderedDict()
        i = 0
        for name, (dtype, shape) in input_metadata.items():
            if input_example is not None and (not isinstance(input_example, tuple) or i < len(input_example)):
                if isinstance(input_example, tuple):
                    static_shape = input_example[i].shape
                elif isinstance(input_example, OrderedDict):
                    static_shape = tuple(input_example.values())[i].shape
                else:
                    static_shape = [tuple(input_example.shape)]
            elif is_shape_dynamic(shape):
                if name in self.default_shapes:
                    static_shape = self.default_shapes[name]
                else:
                    static_shape = [self.default_shape_value if is_dimension_dynamic(elem) else elem for elem in shape]
                if static_shape != shape:
                    if not is_valid_shape_override(static_shape, shape):
                        logging.critical(
                            "Cannot override original shape: {:}, for input: {:} to {:}".format(
                                shape, name, static_shape
                            )
                        )
                    logging.warning(
                        "Input: {:}: Adjusted dynamic shape: {:} to: {:}".format(name, shape, static_shape),
                        mode=logging_mode.ONCE,
                    )
            else:
                if name in self.default_shapes:
                    logging.warning(
                        "Will not override static shape: {:}, for input: {:}".format(shape, name),
                        mode=logging_mode.ONCE,
                    )
                static_shape = shape

            if input_example is not None and (not isinstance(input_example, tuple) or i < len(input_example)):
                if isinstance(input_example, OrderedDict):
                    buffers[name] = list(input_example.values())[i].cpu()
                else:
                    buffers[name] = input_example[i].cpu() if isinstance(input_example, tuple) else input_example.cpu()
            elif np.issubdtype(dtype, np.integer):
                buffers[name] = rng.randint(low=self.int_min, high=self.int_max, size=static_shape, dtype=dtype)
            elif np.issubdtype(dtype, np.bool_):
                buffers[name] = rng.randint(low=0, high=2, size=static_shape).astype(dtype)
            else:
                buffers[name] = (
                    rng.random_sample(size=static_shape) * (self.float_max - self.float_min) + self.float_min
                ).astype(dtype)

            buffers[name] = np.array(
                buffers[name]
            )  # To handle scalars. The above functions return a float if shape is ().

            # If the shape is 1D, and has a length equal to the rank of the provided default shape, it is
            # likely to be a TRT shape tensor, and so should be overriden such that it's value (not shape) is the default shape.
            is_shape_tensor = (
                (not is_shape_dynamic(shape))
                and (name in self.default_shapes)
                and (len(shape) == 1)
                and (shape[0] == len(self.default_shapes[name]))
            )
            if is_shape_tensor:
                buffers[name] = np.array(self.default_shapes[name], dtype=dtype)
                logging.warning(
                    "Assuming {:} is a shape tensor. Setting to: {:}".format(name, buffers[name]),
                    mode=logging_mode.ONCE,
                )
            i = i + 1

        return buffers


# Caches data loaded by a DataLoader for use across multiple runners.
class DataLoaderCache(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cache = {}  # Dict[int, OrderedDict[str, np.ndarray]]

    def load(self, iteration, input_metadata, input_example=None):
        """
        Load the specified iteration from the cache if present, or generate using the data loader.

        Args:
            iteration (int): The iteration whose data to retrieve.
            input_metadata (OrderedDict[str, Tuple[np.dtype, Tuple[int]]]): Input Metadata, including shape and type information. The loader may attempt to match input_metadata when data in the cache does not exactly match a new set of input_metadata.
        """
        if iteration not in self.cache:
            logging.debug("Iteration {:} not found in cache, generating new buffers for all inputs".format(iteration))
            self.cache[iteration] = self.data_loader(iteration, input_metadata, input_example)
            if self.cache[iteration] is None:
                logging.critical(
                    "Received no data from data_loader(iteration, input_metadata) for input_metadata: {:}".format(
                        input_metadata
                    )
                )
        else:
            logging.info("Found iteration {:} in cache".format(iteration))

        feed_dict = OrderedDict()
        for index, (name, (dtype, shape)) in enumerate(input_metadata.items()):
            cached_name = find_in_dict(name, self.cache[iteration], index)
            if cached_name is None:
                logging.warning("Could not find input: {:} in cache, regenerating buffers".format(name))
                self.cache[iteration] = self.data_loader(iteration, input_metadata, input_example)
                cached_name = name

            buffer = self.cache[iteration][cached_name]

            if dtype != buffer.dtype:
                logging.warning(
                    "Cached buffer data type does not match data type for input: {:}. Note: Cached type: {:}, input type: {:}. Attempting to cast".format(
                        name, buffer.dtype, dtype
                    )
                )
                buffer = buffer.astype(dtype)

            if not is_valid_shape_override(buffer.shape, shape):
                logging.warning(
                    "Cached buffer shape does not match shape for input. Note: Cached shape: {:}, input shape: {:}.".format(
                        buffer.shape, shape
                    )
                )
                # Try to permute the shape to match
                try:
                    perm = FormatManager.permutation(
                        FormatManager.deduce_format(buffer.shape), FormatManager.deduce_format(shape)
                    )
                    new_shape = FormatManager.convert(tuple(buffer.shape), FormatManager.deduce_format(shape))
                    logging.warning(
                        "Attempting to permute shape: {:} using permutation {:}. New shape: {:}".format(
                            buffer.shape, perm, new_shape
                        )
                    )
                    buffer = np.transpose(buffer, perm)
                except NotImplementedError as err:
                    # If the FormatManager does not recognize the format, skip permutation.
                    logging.info("Skipping permutation due to {:}".format(err))
                except KeyError as err:
                    # If the FormatManager cannot generate the permutation for the format combination, skip permutation.
                    logging.info("Skipping permutation due to {:}".format(err))

            feed_dict[name] = buffer
        return feed_dict


class BaseModelLoader(object):
    """
    Loads a model for a runner.
    """

    def __call__(self):
        """
        Load the model.

        Returns:
            A model usable by the runner. The return type is dependent on the runner the loader has been implemented for.
        """
        raise NotImplementedError("BaseModelLoader is an abstract class")


class BaseOnnxModelLoader(BaseModelLoader):
    def check(self, model):
        try:
            onnx.checker.check_model(model)
            logging.debug("ONNX Checker Passed")
        except onnx.checker.ValidationError as err:
            logging.warning("ONNX Checker exited with an error: {:}".format(err))
        return model


# ONNX loaders return ONNX models in memory.
class OnnxFileLoader(BaseOnnxModelLoader):
    def __init__(self, path):
        """
        Loads an ONNX model from a file.

        Args:
            path (str): The path from which to load the model.
        """
        self.path = path

    def __call__(self):
        logging.info("Loading {:}".format(self.path))
        return self.check(onnx.load(self.path))

    def __str__(self):
        return "ONNX Model Loader: {:}".format(self.path)

    def __repr__(self):
        return self.__str__()


class OnnxNetworkLoader(BaseModelLoader):
    def __init__(self, onnx_loader, explicit_precision=None):
        """
        Parses an ONNX model to create an engine.

        Args:
            onnx_loader (Callable() -> onnx.ModelProto): A loader that can supply an ONNX model.

        Optional Args:
            explicit_precision (bool): Whether to create the network with explicit precision enabled.
        """
        self.onnx_loader = onnx_loader
        self.explicit_precision = default_value(explicit_precision, False)

    def __call__(self):
        network = TensorRTRunnerV2.create_network(explicit_precision=self.explicit_precision)

        parser = trt.OnnxParser(network, TRT_LOGGER)
        success = parser.parse(self.onnx_loader().SerializeToString())
        if not success:
            for index in range(parser.num_errors):
                logging.error(parser.get_error(index))
            logging.critical("Could not parse ONNX correctly")

        return network, parser


class BuildEngineLoader(BaseModelLoader):
    def __init__(
        self,
        network_loader,
        max_workspace_size=None,
        fp16_mode=None,
        int8_mode=None,
        profile_shapes=None,
        write_engine=None,
        calibrator=None,
        preprocess_network=None,
        layerwise=None,
    ):
        """
        Uses a TensorRT INetworkDefinition to build an engine

        Args:
            network_loader (Callable()->trt.INetworkDefinition): A callable capable of returning an TensorRT INetworkDefinition. The returned network is owned by the BuildEngineLoader and should not be freed manually. The callable may have at most 2 return values if another object needs to be kept alive for the duration of the network, e.g., in the case of a parser. BuildEngineLoader will take ownership of the second return value, and, like the network, it should not be freed by the callable. The first return value must always be the network.

        Optional Args:
            max_workspace_size (int): The maximum workspace size, in bytes, when building the engine.
            fp16_mode (bool): Whether to build the engine with fp16 mode enabled.
            int8_mode (bool): Whether to build the engine with int8 mode enabled.
            profile_shapes (Dict[str, List[shape, shape, shape]]): A mapping of binding name to min/opt/max shapes. Only needed for networks with dynamic input shapes.
            write_engine (str): A directory in which to save the engine.
            calibrator (trt_smeagol.runners.tensorrt_runner_v2.Calibrator): An int8 calibrator. Only required in int8 mode when the network does not have explicit precision.
            preprocess_network (Callable(trt.INetworkDefinition)): Preprocessing function for the network definition. May be used to modify the network after parsing. This is called before enabling layerwise outputs.
            layerwise (bool): Whether to treat the output of every layer as an output of the network. Defaults to False.
        """
        self.network_loader = network_loader
        self.max_workspace_size = default_value(max_workspace_size, 1 << 24)
        self.fp16_mode = default_value(fp16_mode, False)
        self.int8_mode = default_value(int8_mode, False)
        self.profile_shapes = default_value(profile_shapes, OrderedDict())
        self.write_engine = write_engine
        self.written_engine_path = None
        self.calibrator = calibrator
        self.preprocess_network = default_value(preprocess_network, None)
        self.layerwise = default_value(layerwise, False)

    def __call__(self):
        class DummyContextManager(object):
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                return None

        network_parser = self.network_loader()
        try:
            network, parser = network_parser
            assert isinstance(network, trt.INetworkDefinition)
        except (ValueError, AssertionError):
            network = network_parser
            parser = DummyContextManager()

        with trt.Builder(TRT_LOGGER) as builder, network, parser:
            if self.preprocess_network:
                logging.debug("Applying network preprocessing: {:}".format(self.preprocess_network))
                self.preprocess_network(network)

            if self.layerwise:
                TensorRTRunnerV2.mark_layerwise(network)

            if logging.getEffectiveLevel() <= logging.DEBUG:
                TensorRTRunnerV2.log_network(network)

            config = builder.create_builder_config()
            profile = TensorRTRunnerV2.build_profile(builder, network, self.profile_shapes)
            config.add_optimization_profile(profile)

            config.max_workspace_size = int(self.max_workspace_size)
            if self.fp16_mode:
                config.flags = 1 << int(trt.BuilderFlag.FP16)
            if self.int8_mode:
                config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
                if not network.has_explicit_precision:
                    if not self.calibrator:
                        logging.critical(
                            "Network does not have explicit precision. A calibrator must be provided in order to use int8 mode."
                        )
                    self.calibrator.set_input_metadata(get_input_metadata_from_profile(profile, network))
                    config.int8_calibrator = self.calibrator

            logging.debug("Using builder configuration flags: {:}".format(config.flags))
            logging.info(
                "Building engine: max workspace size={:} bytes, fp16={:}, int8={:}, layerwise={:}".format(
                    self.max_workspace_size, self.fp16_mode, self.int8_mode, self.layerwise
                )
            )
            engine = builder.build_engine(network, config)
            self.written_engine_path = write_timestamped(
                contents=lambda: engine.serialize(), dir=self.write_engine, name="tensorrt_runner_v2.engine"
            )
            return engine

    def get_engine_path(self):
        """
        Returns the path at which the engine was written, or None if write_engine was not specified.
        """
        return self.written_engine_path
