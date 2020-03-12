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

# Sets up everything needed to perform inference with TensorRT.
import os
import pickle
import sys
import time
import zlib
from collections import OrderedDict

import numpy as np

# Only initialize GPU after this runner is activated.
import pycuda.autoinit

# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.driver as cuda
import tensorrt as trt

from nemo import logging, logging_mode

logging.info("Using TensorRT {:}".format(trt.__version__))
logging.debug("Note: Using tensorrt from {:}".format(trt.__path__))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def set_trt_logging_level(sev):
    global TRT_LOGGER
    if sev == logging.DEBUG:
        logging.min_severity = trt.Logger.INFO
    elif sev == logging.WARNING:
        logging.min_severity = trt.Logger.WARNING
    elif sev == logging.ERROR:
        logging.min_severity = trt.Logger.ERROR
    elif sev == logging.CRITICAL:
        logging.min_severity = trt.Logger.INTERNAL_ERROR


TRT_DYNAMIC_DIM = -1
DEFAULT_SHAPE_VALUE = 1


# Attempt to partially match output names. Returns None on failure
# Checks for exact matches and substring matches, falling back to index based matching.
def find_in_dict(name, map, index=None):
    if name in map:
        return name
    for key in map.keys():
        if name.lower() in key.lower() or key.lower() in name.lower():
            return key
    if index is not None and index >= 0 and index < len(map.keys()):
        return list(map.keys())[index]
    return None


def default_value(value, default):
    return value if value is not None else default


def is_dimension_dynamic(dim):
    return dim is None or dim <= 0


def is_shape_dynamic(shape):
    return any([is_dimension_dynamic(dim) for dim in shape])


def is_valid_shape_override(new_shape, original_shape):
    ranks_same = len(original_shape) == len(new_shape)
    overrides_valid = all(
        [odim == ndim or is_dimension_dynamic(odim) for odim, ndim in zip(original_shape, new_shape)]
    )
    return ranks_same and overrides_valid


def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


def compress(info):
    return zlib.compress(pickle.dumps(info))


def decompress(bytes):
    return pickle.loads(zlib.decompress(bytes))


def is_compressed(obj):
    return isinstance(obj, bytes)


def is_pickleable(obj):
    try:
        pickle.dumps(obj)
        return True
    except TypeError:
        return False


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def pickle_save(path, obj):
    with open(path, "wb") as f:
        return f.write(pickle.dumps(obj))


# The maximum number of bytes that can be sent at once over a queue.
PIPE_MAX_SEND_BYTES = 1 << 31

# Attempts to send an object over the queue, compresses if needed. In the event the object cannot be sent, sends None instead.
def send_on_queue(queue, obj):
    if not is_pickleable(obj):
        logging.warning("Cannot pickle: {:}. Sending None instead".format(obj))
        queue.put(None)
        return

    if sys.getsizeof(obj) > PIPE_MAX_SEND_BYTES:
        logging.warning(
            "Object size ({:} bytes) exceeds maximum size that can be sent over queues ({:} bytes). Attempting to compress - this may take some time. If this does not work or you want to avoid the compression overhead, you should disable subprocesses via the --no-subprocess flag, or by setting use_subprocess=False in Comparator.run().".format(
                sys.getsizeof(obj), PIPE_MAX_SEND_BYTES
            )
        )
        obj = compress(obj)

    if sys.getsizeof(obj) > PIPE_MAX_SEND_BYTES:
        logging.warning("Compressed object is still too large to send. Sending None instead.")
        queue.put(None)
        return

    logging.info("Sending: {:} on queue".format(obj))
    queue.put(obj)


def receive_on_queue(queue, timeout=None):
    logging.info("Waiting for data to become available on queue")
    obj = queue.get(block=True, timeout=timeout)
    if is_compressed(obj):
        logging.debug("Decompressing output")
        obj = decompress(obj)
    logging.info("Received {:} on queue".format(obj))
    return obj


def timestamped_filepath(dir, name):
    name, ext = os.path.splitext(name)
    return os.path.join(dir, "{:}.{:}{:}".format(name, time.strftime("%Y-%m-%d-%H-%M-%S"), ext))


def write_timestamped(contents, dir=None, name=None, mode="wb"):
    """
    Generates a timestamped file path in the specified directory.

    Args:
        contents (bytes-like object or callable): Either a bytes-like object that can be written to disk, or a callable which will return such an object.
        dir (str): The directory to write into.
        name (str): The name of the file.

    Optional Args:
        mode(str): The mode to use when writing. Defaults to "wb".

    Returns:
        str: The complete file path, or None if nothing was written.
    """
    if dir is not None:
        if not os.path.exists(dir):
            # logging.debug("{:} does not exist, creating now.".format(dir))
            os.makedirs(dir, exist_ok=True)

        path = timestamped_filepath(dir, name)

        if callable(contents):
            contents = contents()

        if os.path.exists(path):
            logging.warning("{:} already exists. Will not overwrite.".format(path))
        else:
            with open(path, mode) as f:
                logging.info("Writing to {:}".format(path))
                f.write(contents)
            return path
    return None


def get_input_metadata_from_profile(profile, network):
    input_metadata = OrderedDict()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.is_shape_tensor:
            shapes = profile.get_shape_input(tensor.name)
        else:
            shapes = profile.get_shape(tensor.name)
        if tuple(shapes[0]) != tuple(shapes[1]):
            logging.warning("In profile 0, min != max, using opt shapes for calibration")
        # Always use opt shape
        input_metadata[tensor.name] = (trt.nptype(tensor.dtype), shapes[1])
    return input_metadata


class BaseBuffer(object):
    def __init__(self, shape=None, dtype=None):
        self.dtype = default_value(dtype, np.float32)
        self.shape = default_value(shape, tuple())
        self.allocate(self.shape)

    # If the new shape is larger, reallocate, otherwise do nothing.
    def resize(self, new_shape):
        if volume(new_shape) > volume(self.shape):
            self.free()
            self.allocate(new_shape)
        self.shape = new_shape

    def __str__(self):
        return "({:}: shape={:}, dtype={:})".format(type(self).__name__, self.shape, self.dtype)

    def __repr__(self):
        return self.__str__()


class DeviceBuffer(BaseBuffer):
    def allocate(self, shape):
        self.ptr = cuda.mem_alloc(volume(shape) * np.dtype(self.dtype).itemsize)

    def free(self):
        self.ptr.free()

    # Copies a numpy buffer to device
    def copy_htod(self, np_buffer, stream=None):
        if stream:
            # PyCUDA requires the host buffer to be pagelocked for asynchronous memcpys.
            pagelocked = cuda.register_host_memory(np.ascontiguousarray(np_buffer.ravel()))
            cuda.memcpy_htod_async(self.ptr, pagelocked, stream)
        else:
            cuda.memcpy_htod(self.ptr, np.ascontiguousarray(np_buffer.ravel()))


class HostBuffer(BaseBuffer):
    def allocate(self, shape):
        self.ptr = cuda.pagelocked_empty(shape, self.dtype).ravel()

    def free(self):
        del self.ptr

    # Copies a DeviceBuffer to host
    def copy_dtoh(self, device_buffer, stream=None):
        if stream:
            cuda.memcpy_dtoh_async(self.ptr, device_buffer.ptr, stream)
        else:
            cuda.memcpy_dtoh(self.ptr, device_buffer.ptr)

    # Return a view of the buffer which has the correct shape
    def view(self):
        return self.ptr[: volume(self.shape)].reshape(self.shape)


class Buffers(object):
    @staticmethod
    def from_engine(engine):
        buffers = Buffers()
        for binding in engine:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            buffers.device_buffers[binding] = DeviceBuffer(dtype=dtype)
            if not engine.binding_is_input(binding):
                buffers.host_outputs[binding] = HostBuffer(dtype=dtype)
        return buffers

    def __init__(self):
        self.device_buffers = OrderedDict()
        self.host_outputs = OrderedDict()

    # Resize the specfied buffer to the specified shape
    def resize(self, name, shape):
        found = False
        for buf_dict in [self.device_buffers, self.host_outputs]:
            if name in buf_dict:
                found = True
                buf_dict[name].resize(shape)

        if not found:
            logging.warning("Buffer: {:} was not found, could not resize".format(name))
        else:
            logging.debug("Resizing {:} buffer to {:}".format(name, shape))

    def copy_inputs(self, feed_dict, stream=None):
        for name, buffer in feed_dict.items():
            self.device_buffers[name].copy_htod(buffer, stream)

    # Copies outputs from the device back to host.
    def copy_outputs(self, stream=None):
        for name, buffer in self.host_outputs.items():
            buffer.copy_dtoh(self.device_buffers[name], stream)

    def get_bindings(self):
        return [int(buf.ptr) for buf in self.device_buffers.values()]

    # Gets a dictionary mapping names to numpy buffers.
    def get_outputs(self):
        out_dict = OrderedDict()
        for name, buffer in self.host_outputs.items():
            out_dict[name] = buffer.view()
        return out_dict

    def free(self):
        [buf.free() for buf in self.device_buffers.values()]
        [buf.free() for buf in self.host_outputs.values()]


class BaseRunner(object):
    def __init__(self, name=None):
        """
        The base class for runner objects. All runners should override the functions and attributes specified here.

        Vars:
            name (str): The name of this runner.
        """
        self.name = default_value(name, "Runner")
        self.inference_time = None

    def __enter__(self):
        """
        Activate the runner for inference. This may involve allocating GPU buffers, for example.

        It is extremely important that the GPU is not used by the runner before the __enter__ function is called.

        Vars:
            inputs (OrderedDict[InputKey, Tuple[int]]): A mapping of input tensor names to their shapes, INCLUDING batch dimension, for this runner. This MUST be known at runner initialization for the Comparator to work correctly. InputKey can be any type used to uniquely indentify an input, e.g. a string containing the input name.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deactivate the runner. All memory allocated by __enter__() should be freed, such that the runner is no longer using the GPU.

        Returns:
            BaseRunner
        """
        pass

    def infer(self, feed_dict, output):
        """
        Runs inference using the provided feed_dict.

        Args:
            feed_dict (OrderedDict[str, np.ndarray]): A mapping of input tensor names to corresponding input numpy arrays.

        Returns:
            OrderedDict[str, np.ndarray]: A mapping of output tensor names to their corresponding numpy arrays.
        """
        raise NotImplementedError("BaseRunner is an abstract class")

    # def last_inference_time(self):
    #     """
    #     Returns the time required for the last call to `infer`
    #
    #     Returns:
    #         float: The time in seconds
    #     """
    #     if self.inference_time is None:
    #         logging.warning("inference_time was not set for this runner. Inference time will be incorrect! To correctly compare runtimes, please set the inference_time property in the infer() function", mode=LogMode.ONCE)
    #         return 1
    #     return self.inference_time

    def get_input_metadata(self, input_examples, output_example):
        """
        Returns information about the inputs of the model. Shapes here may be unknown/dynamic. Must be called after __enter__() and before __exit__()

        Returns:
            OrderedDict[str, Tuple[np.dtype, Tuple[int]]]: Mapping of input names to their data types and shapes.
        """
        raise NotImplementedError("BaseRunner is an abstract class")


# Builds and tracks a single engine for a single network.
class TensorRTRunnerV2(BaseRunner):
    total_runners = 0
    """
    A runner that can perform inference on a single TensorRT engine.
    """

    def __init__(self, model_loader=None, plugins=None, name=None):
        """
        Creates a runner that manages a single TensorRT engine.

        Args:
            model_loader (Callable() -> trt.ICudaEngine): A callable that can supply a TensorRT engine.

        Optional Args:
            max_workspace_size (int): The maximum workspace size in bytes.
            plugins (List[str]): A list of paths to plugin libraries to load before inference.
            name (str): The human-readable name to use for this runner.
        """
        set_trt_logging_level(logging.getEffectiveLevel())

        def load_plugins():
            import ctypes

            for plugin in plugins:
                path = os.path.abspath(plugin)
                logging.info("Loading plugin library: {:}".format(path))
                ctypes.CDLL(path)

        # Load any user-supplied plugin libraries. This must happen before everything else, including engine deserialization.
        if plugins:
            load_plugins()

        # Choose a unique name for this runner.
        super().__init__(default_value(name, "trt-v2-runner-{:}".format(TensorRTRunnerV2.total_runners)))
        TensorRTRunnerV2.total_runners += 1
        logging.debug("Creating {:}".format(self.name))

        self.model_loader = model_loader

        self.engine = self.model_loader()
        if not self.engine:
            logging.critical("Invalid Engine. Please ensure the engine was built correctly.")

        self.buffers = Buffers.from_engine(self.engine)
        self.stream = cuda.Stream()

        self.context = self.engine.create_execution_context()

    def __enter__(self):
        """
        Vars:
            engine (trt.ICudaEngine): The engine tracked by this runner. The TensorRTRunnerV2 OWNS the engine it manages, and therefore is responsible for it's destruction. Do not free the engine outside of the runner, or it will result in a double free.
            context (trt.IExecutionContext): The context used for inference.
            stream (pycuda.driver.Stream): The CUDA stream that this runner will use for inference.
        """
        return self

    @staticmethod
    def override_shape_list(shape):
        return [DEFAULT_SHAPE_VALUE if is_dimension_dynamic(dim) else dim for dim in shape]

    def get_input_metadata(self):
        inputs = OrderedDict()
        active_profile = self.context.active_optimization_profile
        bindings_per_profile = len(self.engine) // self.engine.num_optimization_profiles
        logging.debug(
            "Total # of Profiles: {:}, Bindings Per Profile: {:}, Active Profile: {:}".format(
                self.engine.num_optimization_profiles, bindings_per_profile, active_profile
            )
        )

        start_binding = bindings_per_profile * active_profile
        end_binding = start_binding + bindings_per_profile
        logging.info("Start Binding: {:}, End Binding: {:}".format(start_binding, end_binding))

        for binding in range(start_binding, end_binding):
            if self.engine.binding_is_input(binding):
                inputs[self.engine[binding]] = (
                    trt.nptype(self.engine.get_binding_dtype(binding)),
                    list(self.engine.get_binding_shape(binding)),
                )
        return inputs

    def __exit__(self, exc_type, exc_value, traceback):
        # Destroy the engine, and context.
        with self.engine, self.context:
            pass

        self.buffers.free()
        del self.stream

    def infer(self, feed_dict, output):
        for name in self.engine:
            if name in feed_dict:
                in_out = [feed_dict[name]]
            elif isinstance(output, tuple):
                in_out = [output[i].detach().cpu().numpy() for i in range(len(output))]
            else:
                in_out = [output.detach().cpu().numpy()]

            binding = self.engine[name]

            # Only set shapes if required
            for i in range(len(in_out)):
                shape = in_out[i].shape
                if self.engine.is_shape_binding(binding) and is_shape_dynamic(self.context.get_shape(binding)):
                    logging.debug("Setting shape binding: {:} (index: {:}) to: {:}".format(name, binding, in_out[i]))
                    self.context.set_shape_input(binding, in_out[i])
                elif is_shape_dynamic(self.context.get_binding_shape(binding)):
                    logging.debug("Setting binding: {:} (index: {:}) to shape: {:}".format(name, binding, shape))
                    self.context.set_binding_shape(binding, shape)

        # Check
        if not self.context.all_binding_shapes_specified:
            logging.critical(
                "Some input shapes were not specified.\nNote: Inputs are: {:}".format(self.get_input_metadata())
            )
        if not self.context.all_shape_inputs_specified:
            logging.critical(
                "Some shape inputs were not specified.\nNote: Inputs are: {:}".format(self.get_input_metadata())
            )

        bindings_per_profile = self.engine.num_bindings // self.engine.num_optimization_profiles
        start_binding = self.context.active_optimization_profile * bindings_per_profile
        end_binding = start_binding + bindings_per_profile

        # Resize buffers so they are the appropriate size.
        for binding in range(start_binding, end_binding):
            shape = tuple(self.context.get_binding_shape(binding))
            self.buffers.resize(self.engine[binding], shape)

        bindings = self.buffers.get_bindings()

        start = time.perf_counter()
        self.buffers.copy_inputs(feed_dict, self.stream)
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
        self.buffers.copy_outputs(self.stream)
        self.stream.synchronize()
        end = time.perf_counter()

        self.inference_time = end - start
        return self.buffers.get_outputs()

    # Utility functions related to TensorRT, but not tied to any specific instance.
    @staticmethod
    def create_network(explicit_batch=True, explicit_precision=False):
        with trt.Builder(TRT_LOGGER) as builder:
            network_flags = 0
            if explicit_batch:
                network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            if explicit_precision:
                network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
            network = builder.create_network(flags=network_flags)
            if network is None:
                logging.critical("Invalid network")
            return network

    @staticmethod
    def get_network_inputs(network):
        return {network.get_input(i).name: network.get_input(i).shape for i in range(network.num_inputs)}

    @staticmethod
    def log_network(network):
        LAYER_TYPE_CLASS_MAPPING = {
            trt.LayerType.CONVOLUTION: trt.IConvolutionLayer,
            trt.LayerType.FULLY_CONNECTED: trt.IFullyConnectedLayer,
            trt.LayerType.ACTIVATION: trt.IActivationLayer,
            trt.LayerType.POOLING: trt.IPoolingLayer,
            trt.LayerType.LRN: trt.ILRNLayer,
            trt.LayerType.SCALE: trt.IScaleLayer,
            trt.LayerType.SOFTMAX: trt.ISoftMaxLayer,
            trt.LayerType.DECONVOLUTION: trt.IDeconvolutionLayer,
            trt.LayerType.CONCATENATION: trt.IConcatenationLayer,
            trt.LayerType.ELEMENTWISE: trt.IElementWiseLayer,
            trt.LayerType.PLUGIN: trt.IPluginLayer,
            trt.LayerType.RNN: trt.IRNNLayer,
            trt.LayerType.UNARY: trt.IUnaryLayer,
            trt.LayerType.PADDING: trt.IPaddingLayer,
            trt.LayerType.SHUFFLE: trt.IShuffleLayer,
            trt.LayerType.REDUCE: trt.IReduceLayer,
            trt.LayerType.TOPK: trt.ITopKLayer,
            trt.LayerType.GATHER: trt.IGatherLayer,
            trt.LayerType.MATRIX_MULTIPLY: trt.IMatrixMultiplyLayer,
            trt.LayerType.RAGGED_SOFTMAX: trt.IRaggedSoftMaxLayer,
            trt.LayerType.CONSTANT: trt.IConstantLayer,
            trt.LayerType.RNN_V2: trt.IRNNv2Layer,
            trt.LayerType.IDENTITY: trt.IIdentityLayer,
            trt.LayerType.PLUGIN_V2: trt.IPluginV2Layer,
            trt.LayerType.SLICE: trt.ISliceLayer,
            trt.LayerType.SHAPE: trt.IShapeLayer,
            trt.LayerType.PARAMETRIC_RELU: trt.IParametricReLULayer,
            trt.LayerType.RESIZE: trt.IResizeLayer,
        }

        def is_special_attribute(attr):
            return attr.startswith("__") and attr.endswith("__")

        def is_valid_attribute(attr, layer):
            if (
                type(layer) == trt.IPoolingLayer
                or type(layer) == trt.IConvolutionLayer
                or type(layer) == trt.IDeconvolutionLayer
            ):
                if len(layer.get_input(0).shape) > 4:
                    # 3D pooling uses padding_nd
                    return attr not in ["padding", "stride", "window_size"]
            if type(layer) == trt.IResizeLayer:
                if layer.num_inputs > 1:
                    return attr not in ["scales"]
            if type(layer) == trt.ISliceLayer:
                if layer.num_inputs > 1:
                    return attr not in ["shape", "start", "stride"]
            return True

        logging.debug("Network Inputs: {:}".format(TensorRTRunnerV2.get_network_inputs(network)))
        for layer in network:
            if layer.type in LAYER_TYPE_CLASS_MAPPING:
                layer.__class__ = LAYER_TYPE_CLASS_MAPPING[layer.type]
            input_info = [
                "{:}: {:} ({:})".format(layer.get_input(i).name, layer.get_input(i).shape, layer.get_input(i).dtype)
                for i in range(layer.num_inputs)
                if layer.get_input(i)
            ]
            output_info = [
                "{:}: {:} ({:})".format(layer.get_output(i).name, layer.get_output(i).shape, layer.get_output(i).dtype)
                for i in range(layer.num_outputs)
                if layer.get_output(i)
            ]
            logging.info("{:} [Op: {:}]".format(layer.name, layer.type))
            logging.info("\t{:} -> {:}".format(input_info, output_info))
            attrs = dir(layer)
            for attr in attrs:
                # Exclude special attributes, as well as any attributes of the base layer class (those can be displayed above).
                if (
                    not is_special_attribute(attr)
                    and not hasattr(trt.ILayer, attr)
                    and is_valid_attribute(attr, layer)
                ):
                    logging.info("\t{:}.{:} = {:}".format(layer.name, attr, getattr(layer, attr)))

        network_outputs = {network.get_output(i).name: network.get_output(i).shape for i in range(network.num_outputs)}
        logging.debug("Network Outputs: {:}".format(network_outputs))

    @staticmethod
    def mark_layerwise(network):
        # Layers within loops cannot be marked as network outputs.
        # TODO: FIXME: This assumes that the network is topologically sorted.
        LOOP_START_LAYERS = [trt.LayerType.TRIP_LIMIT, trt.LayerType.ITERATOR]
        LOOP_END_LAYERS = [trt.LayerType.LOOP_OUTPUT]
        num_layers_marked = 0
        in_loop = False
        for layer in network:
            if layer.type in LOOP_START_LAYERS:
                in_loop = True
            elif layer.type in LOOP_END_LAYERS:
                in_loop = False
            for index in range(layer.num_outputs):
                out = layer.get_output(index)
                if not out.is_network_output and not in_loop:
                    logging.debug("Marking {:} as an output".format(out.name))
                    network.mark_output(out)
                    num_layers_marked += 1
        logging.debug("Running in layerwise mode. Marking {:} layers as outputs".format(num_layers_marked))

    @staticmethod
    def build_profile(builder, network, profile_shapes, default_shape_value=DEFAULT_SHAPE_VALUE):
        def override_shape(shape):
            return tuple([DEFAULT_SHAPE_VALUE if is_dimension_dynamic(dim) else dim for dim in shape])

        def get_profile_shape(name):
            if name not in profile_shapes:
                return None
            shapes = profile_shapes[name]
            if not isinstance(shapes, list) or len(shapes) != 3:
                logging.critical(
                    "Profile values must be a list containing exactly 3 shapes (tuples or Dims), but received shapes: {:} for input: {:}.\nNote: profile was: {:}.\nNote: Network inputs were: {:}".format(
                        shapes, name, profile_shapes, TensorRTRunnerV2.get_network_inputs(network)
                    )
                )
            return shapes

        profile = builder.create_optimization_profile()
        for idx in range(network.num_inputs):
            inp = network.get_input(idx)

            if inp.is_shape_tensor:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    rank = inp.shape[0]
                    shapes = [(DEFAULT_SHAPE_VALUE,) * rank] * 3
                    logging.warning(
                        "Setting shape input to {:}. If this is incorrect, for shape input: {:}, please provide tuples for min, opt, and max shapes containing {:} elements".format(
                            shapes[0], inp.name, rank
                        ),
                        mode=logging_mode.ONCE,
                    )
                min, opt, max = shapes
                profile.set_shape(inp.name, min, opt, max)
                inp.shape = opt
                logging.info(
                    "Setting shape input: {:} values to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max)
                )
            else:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    shapes = [override_shape(inp.shape)] * 3
                    logging.warning(
                        "Overriding input shape {:} to {:}. If this is incorrect, for input tensor: {:}, please provide tuples for min, opt, and max shapes containing values: {:} with dynamic dimensions replaced,".format(
                            inp.shape, shapes[0], inp.name, inp.shape
                        ),
                        mode=logging_mode.ONCE,
                    )
                min, opt, max = shapes
                profile.set_shape(inp.name, min, opt, max)
                inp.shape = opt
                logging.info(
                    "Setting input: {:} shape to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max)
                )

        if not profile:
            logging.critical(
                "Profile is not valid, please provide profile data. Note: profile was: {:}".format(profile_shapes)
            )
        return profile
