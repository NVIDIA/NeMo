#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
import ctypes
import json
import os
import os.path
import re
import sys
import time

import numpy as np
import tensorrt as trt

# from helpers.calibrator import BertCalibrator as BertCalibrator

try:
    import torch
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})\n""".format(err))
    sys.exit()

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "1", "")
gelu_plg_creator = plg_registry.get_plugin_creator("CustomGeluPluginDynamic", "1", "")
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "1", "")
fc_plg_creator = plg_registry.get_plugin_creator("CustomFCPluginDynamic", "1", "")


"""
Attentions Keys
"""
WQ = "query_weight"
BQ = "query_bias"
WK = "key_weight"
BK = "key_bias"
WV = "value_weight"
BV = "value_bias"
WQKV = "qkv_weight"
BQKV = "qkv_bias"


"""
Transformer Keys
"""
W_AOUT = "attention_output_dense_weight"
B_AOUT = "attention_output_dense_bias"
AOUT_LN_BETA = "attention_output_layernorm_bias"
AOUT_LN_GAMMA = "attention_output_layernorm_weight"
W_MID = "intermediate_dense_weight"
B_MID = "intermediate_dense_bias"
W_LOUT = "output_dense_weight"
B_LOUT = "output_dense_bias"
LOUT_LN_BETA = "output_layernorm_bias"
LOUT_LN_GAMMA = "output_layernorm_weight"


"""
Squad Output Keys
"""
SQD_W = "weight"
SQD_B = "bias"


class BertConfig:
    def __init__(self, bert_config_path, use_fp16, use_int8, use_strict, use_fc2_gemm):
        with open(bert_config_path, 'r') as f:
            data = json.load(f)
            self.num_attention_heads = data['num_attention_heads']
            self.hidden_size = data['hidden_size']
            self.intermediate_size = data['intermediate_size']
            self.num_hidden_layers = data['num_hidden_layers']
            self.use_fp16 = use_fp16
            self.use_int8 = use_int8
            self.use_fc2_gemm = use_fc2_gemm
            self.use_strict = use_strict
            self.head_size = self.hidden_size // self.num_attention_heads


def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name


def set_output_name(layer, prefix, name, out_idx=0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)


def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    assert len(input_tensor.shape) == 5
    B, S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    # FC_attention
    if config.use_int8:
        mult_all = network.add_convolution(input_tensor, 3 * hidden_size, (1, 1), Wall, Ball)
    else:
        mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)

    set_output_name(mult_all, prefix, "qkv_mult")

    has_mask = imask is not None

    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0)]
    if has_mask:
        qkv_in.append(imask)
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    set_output_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx


def skipln(prefix, config, init_dict, network, input_tensor, skip, bias=None):
    """
    Add the skip layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    pf_ld = trt.PluginField("ld", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32)
    wbeta = init_dict[prefix + "bias"]
    pf_beta = trt.PluginField("beta", wbeta.numpy(), trt.PluginFieldType.FLOAT32)
    wgamma = init_dict[prefix + "weight"]
    pf_gamma = trt.PluginField("gamma", wgamma.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)

    fields = [pf_ld, pf_beta, pf_gamma, pf_type]

    if bias:
        pf_bias = trt.PluginField("bias", bias.numpy(), trt.PluginFieldType.FLOAT32)
        fields.append(pf_bias)

    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    return layer


def my_fc(config, network, input_tensor, out_dims, W):
    pf_out_dims = trt.PluginField('out_dims', np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField('W', W.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.use_fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin('fcplugin', pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense


def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    context_transposed = attention_layer_opt(
        prefix + "attention_self_", config, init_dict, network, input_tensor, imask
    )
    attention_heads = context_transposed.get_output(0)

    # FC0
    B_aout = init_dict[prefix + B_AOUT]
    if config.use_int8:
        W_aout = init_dict[prefix + W_AOUT]
        attention_out_fc = network.add_convolution(attention_heads, hidden_size, (1, 1), W_aout, B_aout)
        B_aout = None

        if config.use_fp16:
            attention_out_fc.precision = trt.DataType.INT8
            attention_out_fc.set_output_type(0, trt.DataType.HALF)
    else:
        W_aoutT = init_dict[prefix + W_AOUT + '_trans']
        attention_out_fc = my_fc(config, network, attention_heads, hidden_size, W_aoutT)

    skiplayer = skipln(
        prefix + "attention_output_layernorm_",
        config,
        init_dict,
        network,
        attention_out_fc.get_output(0),
        input_tensor,
        B_aout,
    )
    attention_ln = skiplayer.get_output(0)

    # FC1 + GELU
    B_mid = init_dict[prefix + B_MID]
    W_mid = init_dict[prefix + W_MID]
    if config.use_int8:
        mid_dense = network.add_convolution(attention_ln, config.intermediate_size, (1, 1), W_mid, B_mid)
    else:
        mid_dense = network.add_fully_connected(attention_ln, config.intermediate_size, W_mid, B_mid)

    mid_dense_out = mid_dense.get_output(0)
    POW = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant((1, 1, 1, 1, 1), trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant(
        (1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32)))
    )
    ONE = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant((1, 1, 1, 1, 1), trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(mid_dense_out, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(mid_dense_out, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), mid_dense_out, trt.ElementWiseOperation.PROD)

    # enable elementwise fusing for int8 && fp16
    POW.precision = trt.DataType.FLOAT
    MULTIPLY.precision = trt.DataType.FLOAT
    SQRT.precision = trt.DataType.FLOAT
    ONE.precision = trt.DataType.FLOAT
    HALF.precision = trt.DataType.FLOAT
    X_pow.precision = trt.DataType.FLOAT
    X_mul.precision = trt.DataType.FLOAT
    X_add.precision = trt.DataType.FLOAT
    X_sqrt.precision = trt.DataType.FLOAT
    X_tanh.precision = trt.DataType.FLOAT
    X_one.precision = trt.DataType.FLOAT
    CDF.precision = trt.DataType.FLOAT
    gelu_layer.precision = trt.DataType.FLOAT

    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")
    if config.use_int8 and config.use_strict:
        intermediate_act.set_dynamic_range(-10, 10)

    # FC2
    # Dense to hidden size
    B_lout = init_dict[prefix + B_LOUT]
    if config.use_int8 and config.use_strict and not config.use_fc2_gemm:
        W_lout = init_dict[prefix + W_LOUT]
        out_dense = network.add_convolution(intermediate_act, hidden_size, (1, 1), W_lout, B_lout)
        B_lout = None
    else:
        W_loutT = init_dict[prefix + W_LOUT + '_trans']
        out_dense = my_fc(config, network, intermediate_act, hidden_size, W_loutT)

    set_output_name(out_dense, prefix + "output_", "dense")
    out_layer = skipln(
        prefix + "output_layernorm_", config, init_dict, network, out_dense.get_output(0), attention_ln, B_lout
    )
    out_ln = out_layer.get_output(0)

    set_tensor_name(out_ln, prefix + "output_", "reshape")

    return out_ln


def bert_model(config, init_dict, network, input_tensor, input_mask):
    """
    Create the bert model
    """
    prev_input = input_tensor
    for layer in range(0, config.num_hidden_layers):
        ss = "l{}_".format(layer)
        prev_input = transformer_layer_opt(ss, config, init_dict, network, prev_input, input_mask)
    return prev_input


def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)

    OUT = network.add_shuffle(dense.get_output(0))
    OUT.second_transpose = (1, 0, 2, 3, 4)
    return OUT


def sequence_class_output(prefix, init_dict, network, input_tensor, softmax=True):
    # (seq_len, batch, hidden size, 1, 1)
    hidden_size = input_tensor.shape[2]

    shuf = network.add_shuffle(input_tensor)
    shuf.first_transpose = (1, 3, 4, 0, 2)  # target = (batch, 1, 1, seq_len, hidden_size)

    in_shape_tensor = network.add_shape(shuf.get_output(0)).get_output(0)
    out_shape_tensor = network.add_gather(
        in_shape_tensor,
        network.add_constant((5,), trt.Weights(np.array([0, 1, 2, 2, 4]).astype(np.int32))).get_output(0),
        0,
    ).get_output(0)

    first_token_tensor = network.add_slice(
        shuf.get_output(0), start=(0, 0, 0, 0, 0), shape=(-1, 1, 1, 1, hidden_size), stride=(1, 1, 1, 1, 1),
    )
    first_token_tensor.set_input(
        1, network.add_constant((5,), trt.Weights(np.array([0, 0, 0, 0, 0]).astype(np.int32))).get_output(0),
    )
    first_token_tensor.set_input(2, out_shape_tensor)

    W_out = init_dict[prefix + "mlp.layer0." + SQD_W]
    B_out = init_dict[prefix + "mlp.layer0." + SQD_B]
    dense = network.add_fully_connected(first_token_tensor.get_output(0), W_out.shape[0], W_out, B_out)
    dense_relu = network.add_activation(dense.get_output(0), trt.ActivationType.RELU)
    W_out = init_dict[prefix + "mlp.layer2." + SQD_W]
    B_out = init_dict[prefix + "mlp.layer2." + SQD_B]
    classifier = network.add_fully_connected(dense_relu.get_output(0), W_out.shape[0], W_out, B_out)
    if softmax:
        probs = network.add_softmax(classifier.get_output(0))
        probs.axes = 4  # last dimension
        classifier = probs
    classifier = network.add_shuffle(classifier.get_output(0))
    classifier.reshape_dims = trt.Dims([0, W_out.shape[0]])

    set_output_name(classifier, prefix, "classifier")
    return classifier


def token_class_output(prefix, init_dict, network, input_tensor, softmax=True):
    W_out = init_dict[prefix + "mlp.layer0." + SQD_W]
    B_out = init_dict[prefix + "mlp.layer0." + SQD_B]
    dense = network.add_fully_connected(input_tensor, W_out.shape[0], W_out, B_out)
    dense_relu = network.add_activation(dense.get_output(0), trt.ActivationType.RELU)
    W_out = init_dict[prefix + "mlp.layer2." + SQD_W]
    B_out = init_dict[prefix + "mlp.layer2." + SQD_B]
    classifier = network.add_fully_connected(dense_relu.get_output(0), W_out.shape[0], W_out, B_out)

    if softmax:
        probs = network.add_softmax(classifier.get_output(0))
        probs.axes = 4  # last dimension
        classifier = probs
    set_output_name(classifier, prefix, "classifier")
    classifier = network.add_shuffle(classifier.get_output(0))
    classifier.reshape_dims = trt.Dims([0, 0, 0])
    classifier.second_transpose = (1, 0, 2, 3, 4)
    return classifier


def load_weights(inputbase, config):
    """
    Load the weights from the tensorflow checkpoint
    """
    weights_dict = dict()

    try:
        # reader = pyTF.NewCheckpointReader(inputbase)
        tensor_dict = torch.load(inputbase, map_location='cpu')

        # There might be training-related variables in the checkpoint that
        # can be discarded
        param_names = [key for key in sorted(tensor_dict) if 'adam' not in key and 'global_step' not in key]
        count = len(param_names)
        TRT_LOGGER.log(TRT_LOGGER.INFO, f"Loading/transforming {str(count)} weights")

        for pn in param_names:
            toks = pn.lower().split('.')
            if 'encoder' in pn:
                assert 'layer' in pn
                lvar = (re.findall('\d+', pn))[0]  # nopep8
                outname = 'l{}_'.format(lvar) + '_'.join(toks[4:])
            else:
                outname = '_'.join(toks)

            tensor = tensor_dict[pn].numpy()
            if pn.find('weight') != -1:
                weights_dict[outname + '_trans'] = trt.Weights(np.ascontiguousarray(np.transpose(tensor)).flatten())
                TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Transposing {}\n".format(pn))

            # convert torch tensor to numpy
            shape = tensor.shape
            flat_tensor = tensor.flatten()
            shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
            weights_dict[outname] = trt.Weights(flat_tensor)

            TRT_LOGGER.log(
                TRT_LOGGER.VERBOSE, "Orig.name: {:}, TRT name: {:}, shape: {:}".format(pn, outname, shape_str),
            )

        N = config.num_attention_heads
        H = config.head_size

        additional_dict = dict()
        for key, value in weights_dict.items():
            pos = key.find(BQ)
            if pos != -1:
                hidden_size = value.size
                prefix = key[:pos]

                Bq_ = value
                Bk_ = weights_dict[prefix + BK]
                Bv_ = weights_dict[prefix + BV]
                Wq_ = weights_dict[prefix + WQ]
                Wk_ = weights_dict[prefix + WK]
                Wv_ = weights_dict[prefix + WV]

                mat_size = hidden_size * hidden_size
                wcount = 3 * mat_size
                Wall = np.zeros(wcount, np.float32)
                bcount = 3 * hidden_size
                Ball = np.zeros(bcount, np.float32)
                Wall[0:mat_size] = Wq_.numpy()[0:mat_size]
                Wall[mat_size : 2 * mat_size] = Wk_.numpy()[0:mat_size]
                Wall[2 * mat_size : 3 * mat_size] = Wv_.numpy()[0:mat_size]
                Ball[0:hidden_size] = Bq_.numpy()[0:hidden_size]
                Ball[hidden_size : 2 * hidden_size] = Bk_.numpy()[0:hidden_size]
                Ball[2 * hidden_size : 3 * hidden_size] = Bv_.numpy()[0:hidden_size]

                Wall = np.ascontiguousarray(Wall.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)), dtype=np.float32)
                Ball = np.ascontiguousarray(Ball.reshape((3, N, H)).transpose((1, 0, 2)), dtype=np.float32)

                additional_dict[prefix + WQKV] = trt.Weights(Wall)
                additional_dict[prefix + BQKV] = trt.Weights(Ball)

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    weights_dict.update(additional_dict)
    return weights_dict


def emb_layernorm(
    builder,
    network,
    config,
    weights_dict,
    builder_config,
    sequence_length,
    batch_size,
    min_batch_size=None,
    max_batch_size=None,
):
    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1, sequence_length))
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1, sequence_length))
    input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(-1, sequence_length))

    profile = builder.create_optimization_profile()
    min_shape = (min_batch_size or batch_size, sequence_length)
    shape = (batch_size, sequence_length)
    max_shape = (max_batch_size or batch_size, sequence_length)
    profile.set_shape("input_ids", min=min_shape, opt=shape, max=max_shape)
    profile.set_shape("segment_ids", min=min_shape, opt=shape, max=max_shape)
    profile.set_shape("input_mask", min=min_shape, opt=shape, max=max_shape)
    builder_config.add_optimization_profile(profile)

    input_ids_t = network.add_shuffle(input_ids)
    input_ids_t.second_transpose = (1, 0)
    segment_ids_t = network.add_shuffle(segment_ids)
    segment_ids_t.second_transpose = (1, 0)
    input_mask_t = network.add_shuffle(input_mask)
    input_mask_t.second_transpose = (1, 0)

    wbeta = trt.PluginField(
        "bert_embeddings_layernorm_beta",
        weights_dict["bert_embeddings_layernorm_bias"].numpy(),
        trt.PluginFieldType.FLOAT32,
    )
    wgamma = trt.PluginField(
        "bert_embeddings_layernorm_gamma",
        weights_dict["bert_embeddings_layernorm_weight"].numpy(),
        trt.PluginFieldType.FLOAT32,
    )
    wwordemb = trt.PluginField(
        "bert_embeddings_word_embeddings",
        weights_dict["bert_embeddings_word_embeddings_weight"].numpy(),
        trt.PluginFieldType.FLOAT32,
    )
    wtokemb = trt.PluginField(
        "bert_embeddings_token_type_embeddings",
        weights_dict["bert_embeddings_token_type_embeddings_weight"].numpy(),
        trt.PluginFieldType.FLOAT32,
    )
    wposemb = trt.PluginField(
        "bert_embeddings_position_embeddings",
        weights_dict["bert_embeddings_position_embeddings_weight"].numpy(),
        trt.PluginFieldType.FLOAT32,
    )

    output_fp16 = trt.PluginField(
        "output_fp16", np.array([1 if config.use_fp16 else 0]).astype(np.int32), trt.PluginFieldType.INT32
    )

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
    fn = emln_plg_creator.create_plugin("embeddings", pfc)

    inputs = [input_ids_t.get_output(0), segment_ids_t.get_output(0), input_mask_t.get_output(0)]
    emb_layer = network.add_plugin_v2(inputs, fn)
    set_output_name(emb_layer, "embeddings_", "output")
    return emb_layer


def build_engine(
    batch_size,
    sequence_length,
    config,
    weights_dict,
    classifiers_dict,
    squad_json,
    vocab_file,
    calibrationCacheFile,
    calib_num,
    tok_class_prefix=None,
    seq_class_prefix=None,
    qa_prefix=None,
    min_batch_size=None,
    max_batch_size=None,
):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        explicit_batch_flag
    ) as network, builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = 5000 * (1024 * 1024)  # 5000 MiB
        if config.use_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.use_int8:
            calibrator = BertCalibrator(squad_json, vocab_file, calibrationCacheFile, 1, sequence_length, calib_num)
            builder_config.set_flag(trt.BuilderFlag.INT8)
            builder_config.int8_calibrator = calibrator
        if config.use_strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Create the network
        emb_layer = emb_layernorm(
            builder,
            network,
            config,
            weights_dict,
            builder_config,
            sequence_length,
            batch_size,
            min_batch_size,
            max_batch_size,
        )
        embeddings = emb_layer.get_output(0)
        mask_idx = emb_layer.get_output(1)

        bert_out = bert_model(config, weights_dict, network, embeddings, mask_idx)
        if not classifiers_dict:
            classifiers_dict = weights_dict

        if tok_class_prefix is not None:
            TRT_LOGGER.log(TRT_LOGGER.INFO, f"Configuring head for token classification: {tok_class_prefix}")
            token_class = token_class_output(tok_class_prefix, classifiers_dict, network, bert_out)
            token_class_logits_out = token_class.get_output(0)
            network.mark_output(token_class_logits_out)
            token_class_logits_out.name = "token_logits"
            token_class_logits_out.dtype = trt.DataType.FLOAT

        if seq_class_prefix is not None:
            TRT_LOGGER.log(TRT_LOGGER.INFO, f"Configuring head for sequence classification: {seq_class_prefix}")
            seq_class = sequence_class_output(seq_class_prefix, classifiers_dict, network, bert_out)
            seq_class_logits_out = seq_class.get_output(0)
            network.mark_output(seq_class_logits_out)
            seq_class_logits_out.name = "seq_logits"
            seq_class_logits_out.dtype = trt.DataType.FLOAT

        if qa_prefix is not None:
            TRT_LOGGER.log(TRT_LOGGER.INFO, f"Configuring head for question answering: {qa_prefix}")
            qa_logits = squad_output(qa_prefix, config, classifiers_dict, network, bert_out)
            qa_logits_out = qa_logits.get_output(0)
            network.mark_output(qa_logits_out)
            qa_logits_out.name = "qa_logits"
            qa_logits_out.dtype = trt.DataType.FLOAT

        build_start_time = time.time()
        TRT_LOGGER.log(TRT_LOGGER.INFO, f"Starting engine build")
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = time.time() - build_start_time
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Built engine in {:.3f} Sec".format(build_time_elapsed))
        if config.use_int8:
            calibrator.free()
        return engine


def generate_calibration_cache(sequence_length, config, weights_dict, squad_json, vocab_file, calib_num):
    # dynamic shape not working with calibration, so we need generate a calibration cache first using fulldims network
    calibrationCacheFile = "bertSquadCalibCache"
    if not config.use_int8 or os.path.exists(calibrationCacheFile):
        return calibrationCacheFile

    # generate calibration cache
    saved_use_fp16 = config.use_fp16
    config.use_fp16 = False

    # with build_engine([1], sequence_length, config, weights_dict, squad_json, vocab_file, calibrationCacheFile, calib_num) as engine:
    #    TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibrationCacheFile))

    config.use_fp16 = saved_use_fp16
    return calibrationCacheFile


def main():
    parser = argparse.ArgumentParser(
        description='TensorRT BERT Sample', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-bw', '--bert-weight', required=True, help='bert weight from nemo')
    parser.add_argument(
        '-cw', '--class-weight', required=False, default=None, help='classifier weight from nemo',
    )
    parser.add_argument(
        '-t', '--token-classifier', required=False, default=None, help="Name of the token classifier",
    )
    parser.add_argument(
        '-s', '--seq-classifier', required=False, default=None, help="Name of the sequence classifier",
    )
    parser.add_argument(
        '-qa', '--qa', required=False, default=None, help="Name of the Question Answering classifier",
    )
    parser.add_argument(
        '-o', '--output', required=True, help='The bert engine file, ex bert.engine',
    )
    parser.add_argument(
        '--min-batch-size',
        type=int,
        required=False,
        default=None,
        help='Minimum batch size (default = same as ' 'batch-size)',
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        required=False,
        default=None,
        help='Maximum batch size (default = same as ' 'batch-size)',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=1,
        type=int,
        required=False,
        help='Batch size to optimize for. The engine will be usable with any batch size below this, but may not be optimal for smaller sizes.',
    )
    parser.add_argument('-l', '--sequence-length', default=128, help='Sequence length of the BERT model', type=int)
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google',
    )
    parser.add_argument(
        '-f',
        '--no-fp16',
        action='store_true',
        help='Indicates that inference should be run in FP16 precision',
        required=False,
    )
    parser.add_argument(
        '-i',
        '--int8',
        action='store_true',
        help='Indicates that inference should be run in INT8 precision',
        required=False,
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Indicates that inference should be run in strict precision mode',
        required=False,
    )
    parser.add_argument(
        '-j',
        '--squad-json',
        default='squad/dev-v1.1.json',
        help='squad json dataset used for int8 calibration',
        required=False,
    )
    parser.add_argument(
        '-v',
        '--vocab-file',
        default='./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt',
        help='Path to file containing entire understandable vocab',
        required=False,
    )
    parser.add_argument('-n', '--calib-num', default=100, help='calibration batch numbers', type=int)
    parser.add_argument(
        '-g', '--force-fc2-gemm', action='store_true', help='Force use gemm to implement FC2 layer', required=False
    )

    args, _ = parser.parse_known_args()

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Using configuration file: {:}".format(args.config))
    config = BertConfig(args.config, not args.no_fp16, args.int8, args.strict, args.force_fc2_gemm)

    weights_dict = load_weights(args.bert_weight, config)
    classifiers_dict = None
    if args.class_weight:
        classifiers_dict = {k: v.numpy() for k, v in torch.load(args.class_weight, map_location='cpu').items()}

    # return
    calib_cache = generate_calibration_cache(
        args.sequence_length, config, weights_dict, args.squad_json, args.vocab_file, args.calib_num
    )

    with build_engine(
        args.batch_size,
        args.sequence_length,
        config,
        weights_dict,
        classifiers_dict,
        args.squad_json,
        args.vocab_file,
        calib_cache,
        args.calib_num,
        tok_class_prefix=args.token_classifier,
        seq_class_prefix=args.seq_classifier,
        qa_prefix=args.qa,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
    ) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, 'wb') as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


if __name__ == '__main__':
    main()
