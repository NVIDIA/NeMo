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
import re

import numpy as np
import tensorrt as trt
import torch

from nemo import logging

nvinfer = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
cm = ctypes.CDLL("libcommon.so", mode=ctypes.RTLD_GLOBAL)
pg = ctypes.CDLL("libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)

"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()
qkv2_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "1", "")
skln_plg_creator = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "1", "")
gelu_plg_creator = plg_registry.get_plugin_creator("CustomGeluPluginDynamic", "1", "")
emln_plg_creator = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "1", "")

logging.info(
    "creators:", plg_registry, qkv2_plg_creator, skln_plg_creator, gelu_plg_creator, emln_plg_creator,
)
logging.info("\n".join([x.name for x in plg_registry.plugin_creator_list]))

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

# Pooler Keys
POOL_W = "pooler_dense_weight"
POOL_B = "pooler_dense_bias"

# classifier Output Keys
SQD_W = "weight"
SQD_B = "bias"


class BertConfig:
    def __init__(self, bert_config_path):
        with open(bert_config_path, 'r') as f:
            data = json.load(f)
            self.num_attention_heads = data['num_attention_heads']
            self.hidden_size = data['hidden_size']
            self.intermediate_size = data['intermediate_size']
            self.num_hidden_layers = data['num_hidden_layers']
            self.use_fp16 = True


def set_tensor_name(tensor, prefix, name):
    tensor.name = prefix + name


def set_layer_name(layer, prefix, name, out_idx=0):
    set_tensor_name(layer.get_output(out_idx), prefix, name)


def attention_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the attention layer
    """
    assert len(input_tensor.shape) == 5
    B, S, hidden_size, _, _ = input_tensor.shape
    num_heads = config.num_attention_heads
    head_size = int(hidden_size / num_heads)

    Wall = init_dict[prefix + WQKV]
    Ball = init_dict[prefix + BQKV]

    mult_all = network.add_fully_connected(input_tensor, 3 * hidden_size, Wall, Ball)
    set_layer_name(mult_all, prefix, "qkv_mult")

    has_mask = imask is not None

    pf_hidden_size = trt.PluginField("hidden_size", np.array([hidden_size], np.int32), trt.PluginFieldType.INT32,)
    pf_num_heads = trt.PluginField("num_heads", np.array([num_heads], np.int32), trt.PluginFieldType.INT32)
    pf_S = trt.PluginField("S", np.array([S], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([has_mask], np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_S, pf_has_mask])
    qkv2ctx_plug = qkv2_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv_in = [mult_all.get_output(0), imask]
    qkv2ctx = network.add_plugin_v2(qkv_in, qkv2ctx_plug)
    set_layer_name(qkv2ctx, prefix, "context_layer")
    return qkv2ctx


def skipln(prefix, init_dict, network, input_tensor, skip):
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

    pfc = trt.PluginFieldCollection([pf_ld, pf_beta, pf_gamma])
    skipln_plug = skln_plg_creator.create_plugin("skipln", pfc)

    skipln_inputs = [input_tensor, skip]
    layer = network.add_plugin_v2(skipln_inputs, skipln_plug)
    layer.name = prefix + 'skiplayer'
    return layer


def transformer_layer_opt(prefix, config, init_dict, network, input_tensor, imask):
    """
    Add the transformer layer
    """
    idims = input_tensor.shape
    assert len(idims) == 5
    hidden_size = idims[2]

    context_transposed = attention_layer_opt(
        prefix + "attention_self_", config, init_dict, network, input_tensor, imask,
    )
    attention_heads = context_transposed.get_output(0)

    W_aout = init_dict[prefix + W_AOUT]
    B_aout = init_dict[prefix + B_AOUT]
    attention_out_fc = network.add_fully_connected(attention_heads, hidden_size, W_aout, B_aout)

    skiplayer = skipln(
        prefix + "attention_output_layernorm_", init_dict, network, attention_out_fc.get_output(0), input_tensor,
    )
    attention_ln = skiplayer.get_output(0)

    W_mid = init_dict[prefix + W_MID]
    B_mid = init_dict[prefix + B_MID]
    mid_dense = network.add_fully_connected(attention_ln, config.intermediate_size, W_mid, B_mid)

    mid_dense_out = mid_dense.get_output(0)

    pfc = trt.PluginFieldCollection()
    plug = gelu_plg_creator.create_plugin("gelu", pfc)

    gelu_layer = network.add_plugin_v2([mid_dense_out], plug)

    intermediate_act = gelu_layer.get_output(0)
    set_tensor_name(intermediate_act, prefix, "gelu")

    # Dense to hidden size
    W_lout = init_dict[prefix + W_LOUT]
    B_lout = init_dict[prefix + B_LOUT]

    out_dense = network.add_fully_connected(intermediate_act, hidden_size, W_lout, B_lout)
    set_layer_name(out_dense, prefix + "output_", "dense")
    out_layer = skipln(prefix + "output_layernorm_", init_dict, network, out_dense.get_output(0), attention_ln,)
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


# first_token_tensor = hidden_states[:, 0]
# pooled_output = self.dense(first_token_tensor), nn.Linear(
# config.hidden_size, config.hidden_size)
# pooled_output = self.activation(pooled_output), nn.tanh


def bert_pooler(prefix, init_dict, network, input_tensor):
    """
    pooler the bert output
    """
    seq_len = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1]

    shuf = network.add_shuffle(input_tensor)
    shuf.first_transpose = (2, 3, 0, 1)

    first_token_tensor = network.add_slice(
        shuf.get_output(0), start=(0, 0, 0, 0), shape=(1, 1, 1, hidden_size), stride=(1, 1, 1, 1),
    )

    W_out = init_dict[prefix + POOL_W]
    B_out = init_dict[prefix + POOL_B]
    pooler = network.add_fully_connected(first_token_tensor.get_output(0), hidden_size, W_out, B_out)

    pooler = network.add_activation(pooler.get_output(0), trt.ActivationType.TANH)
    set_layer_name(pooler, prefix, "pooler")

    return pooler.get_output(0)


def squad_output(prefix, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 5

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)
    set_layer_name(dense, prefix, "dense")
    return dense


def sequence_class_output(prefix, init_dict, network, input_tensor, softmax=True):
    logging.info(input_tensor.shape)
    seq_len = input_tensor.shape[1]
    hidden_size = input_tensor.shape[2]

    shuf = network.add_shuffle(input_tensor)
    shuf.first_transpose = (0, 3, 4, 1, 2)
    logging.info("seq class in: ", shuf.get_output(0).shape)

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
    classifier.reshape_dims = trt.Dims([0, -1])

    set_layer_name(classifier, prefix, "classifier")
    logging.info("seq class: ", classifier.get_output(0).shape)
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
    set_layer_name(classifier, prefix, "classifier")
    classifier = network.add_shuffle(classifier.get_output(0))
    classifier.reshape_dims = trt.Dims([0, 0, 0])

    logging.info("tok class: ", classifier.get_output(0).shape)
    return classifier


def load_weights(inputbase):
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
        TRT_LOGGER.log(TRT_LOGGER.INFO, str(count))

        for pn in param_names:
            toks = pn.lower().split('.')
            if 'encoder' in pn:
                assert 'layer' in pn
                lvar = (re.findall('\d+', pn))[0]  # nopep8
                outname = 'l{}_'.format(lvar) + '_'.join(toks[4:])
            else:
                outname = '_'.join(toks)

            # convert torch tensor to numpy
            tensor = tensor_dict[pn].numpy()
            shape = tensor.shape
            flat_tensor = tensor.flatten()
            shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
            weights_dict[outname] = trt.Weights(flat_tensor)

            TRT_LOGGER.log(
                TRT_LOGGER.INFO, "Orig.name: {:}, TRT name: {:}, shape: {:}".format(pn, outname, shape_str),
            )

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

                additional_dict[prefix + WQKV] = trt.Weights(Wall)
                additional_dict[prefix + BQKV] = trt.Weights(Ball)

    except Exception as error:
        TRT_LOGGER.log(TRT_LOGGER.ERROR, str(error))

    weights_dict.update(additional_dict)
    return weights_dict


def main(
    bert_weight_path,
    class_weight_path,
    B,
    S,
    config_path,
    outputbase,
    min_batch=None,
    max_batch=None,
    seq_class_prefix=None,
    tok_class_prefix=None,
    qa_prefix=None,
):
    bert_config_path = config_path
    TRT_LOGGER.log(TRT_LOGGER.INFO, bert_config_path)
    config = BertConfig(bert_config_path)

    # Load weights from checkpoint file
    init_dict = load_weights(bert_weight_path)
    classifiers_dict = {k: v.numpy() for k, v in torch.load(class_weight_path, map_location='cpu').items()}

    #    import pdb;pdb.set_trace()
    with trt.Builder(TRT_LOGGER) as builder:
        ty = trt.PluginFieldType.FLOAT32

        # import pdb;pdb.set_trace()
        w = init_dict["bert_embeddings_layernorm_bias"]
        wbeta = trt.PluginField("bert_embeddings_layernorm_beta", w.numpy(), ty)

        w = init_dict["bert_embeddings_layernorm_weight"]
        wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", w.numpy(), ty)

        w = init_dict["bert_embeddings_word_embeddings_weight"]
        wwordemb = trt.PluginField("bert_embeddings_word_embeddings", w.numpy(), ty)

        w = init_dict["bert_embeddings_token_type_embeddings_weight"]
        wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", w.numpy(), ty)

        w = init_dict["bert_embeddings_position_embeddings_weight"]
        wposemb = trt.PluginField("bert_embeddings_position_embeddings", w.numpy(), ty)

        pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb])
        fn = emln_plg_creator.create_plugin("embeddings", pfc)

        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with builder.create_network(explicit_batch_flag) as network, builder.create_builder_config() as builder_config:
            builder_config.max_workspace_size = 5000 * (1024 * 1024)  # 5000 MiB
            builder_config.set_flag(trt.BuilderFlag.FP16)

            input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1, S,))
            segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=(-1, S,))
            input_mask = network.add_input(name="input_mask", dtype=trt.int32, shape=(-1, S,))

            def set_profile_shape(profile, batch_size, min_batch=None, max_batch=None):
                opt_shape = (batch_size, S)
                min_shape = (min_batch or batch_size, S)
                max_shape = (max_batch or batch_size, S)
                profile.set_shape("input_ids", min=min_shape, opt=opt_shape, max=max_shape)
                profile.set_shape("segment_ids", min=min_shape, opt=opt_shape, max=max_shape)
                profile.set_shape("input_mask", min=min_shape, opt=opt_shape, max=max_shape)

            # Specify only a single profile for now, even though this is
            # less optimal
            bs1_profile = builder.create_optimization_profile()
            set_profile_shape(bs1_profile, B, min_batch=min_batch, max_batch=max_batch)
            builder_config.add_optimization_profile(bs1_profile)

            inputs = [input_ids, segment_ids, input_mask]
            emb_layer = network.add_plugin_v2(inputs, fn)

            embeddings = emb_layer.get_output(0)
            mask_idx = emb_layer.get_output(1)

            bert_out = bert_model(config, init_dict, network, embeddings, mask_idx)

            if tok_class_prefix is not None:
                token_class = token_class_output(tok_class_prefix, classifiers_dict, network, bert_out)
                token_class_logits_out = token_class.get_output(0)
                token_class_logits_out.name = "token_logits"
                token_class_logits_out.dtype = trt.DataType.FLOAT
                network.mark_output(token_class_logits_out)

            if seq_class_prefix is not None:
                seq_class = sequence_class_output(seq_class_prefix, classifiers_dict, network, bert_out)
                seq_class_logits_out = seq_class.get_output(0)
                seq_class_logits_out.name = "seq_logits"
                seq_class_logits_out.dtype = trt.DataType.FLOAT
                network.mark_output(seq_class_logits_out)

            if qa_prefix is not None:
                qa_logits = squad_output(seq_class_prefix, classifiers_dict, network, bert_out)
                qa_logits_out = qa_logits.get_output(0)
                qa_logits_out.name = "qa_logits"
                qa_logits_out.dtype = trt.DataType.FLOAT
                network.mark_output(qa_logits_out)

            with builder.build_engine(network, builder_config) as engine:
                TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
                serialized_engine = engine.serialize()
                TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(outputbase))
                with open(outputbase, 'wb') as fout:
                    fout.write(serialized_engine)
                TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TensorRT BERT Sample')
    parser.add_argument('-bw', '--bert-weight', required=True, help='bert weight from nemo')
    parser.add_argument(
        '-cw', '--class-weight', required=True, help='classifier weight from nemo',
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
        '-b', '--batch-size', type=int, required=False, default=1, help='Preferred batch size (default = 1)',
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        required=False,
        default=None,
        help='Maximum batch size (default = same as ' 'batch-size)',
    )
    parser.add_argument(
        '--min-batch-size',
        type=int,
        required=False,
        default=None,
        help='Minimum batch size (default = same as ' 'batch-size)',
    )

    parser.add_argument(
        '-l',
        '--seq-length',
        type=int,
        required=False,
        default=128,
        help='Sequence length of the BERT model (default=128)',
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='The folder containing the bert_config.json, '
        'which can be downloaded e.g. from '
        'https://github.com/google-research/bert#pre'
        '-trained-models or by running '
        'download_models.py in '
        'dle/TensorFlow/LanguageModeling/BERT/'
        'data/pretrained_models_google',
    )

    opt = parser.parse_args()

    outputbase = opt.output
    config_path = opt.config
    logging.info("token class:", opt.token_classifier)
    logging.info("seq class:  ", opt.seq_classifier)
    logging.info("QA class:  ", opt.qa)
    main(
        opt.bert_weight,
        opt.class_weight,
        opt.batch_size,
        opt.seq_length,
        config_path,
        outputbase,
        min_batch=opt.min_batch_size,
        max_batch=opt.max_batch_size,
        tok_class_prefix=opt.token_classifier,
        seq_class_prefix=opt.seq_classifier,
        qa_prefix=opt.qa
    )
