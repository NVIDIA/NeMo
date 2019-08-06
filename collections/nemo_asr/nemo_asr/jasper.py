# Copyright (c) 2019 NVIDIA Corporation
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from .parts.jasper import JasperBlock, jasper_activations, init_weights


class JasperEncoder(TrainableNM):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu"].
        feat_in (int): Number of channels being input to this module
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add" or "max".
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "audio_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(SpectrogramSignalTag),
                                        2: AxisType(ProcessedTimeTag)}),
            "length": NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "outputs": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(EncodedRepresentationTag),
                2: AxisType(ProcessedTimeTag),
            }),

            "encoded_lengths": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            jasper,
            activation,
            feat_in,
            normalization_mode="batch",
            residual_mode="add",
            norm_groups=-1,
            conv_mask=True,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        TrainableNM.__init__(self, **kwargs)

        activation = jasper_activations[activation]()
        feat_in = feat_in * frame_splicing

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            tied = lcfg.get('tied', False)
            heads = lcfg.get('heads', -1)
            encoder_layers.append(
                JasperBlock(feat_in,
                            lcfg['filters'],
                            repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'],
                            stride=lcfg['stride'],
                            dilation=lcfg['dilation'],
                            dropout=lcfg['dropout'],
                            residual=lcfg['residual'],
                            groups=groups,
                            separable=separable,
                            heads=heads,
                            residual_mode=residual_mode,
                            normalization=normalization_mode,
                            norm_groups=norm_groups,
                            tied=tied,
                            activation=activation,
                            residual_panes=dense_res,
                            conv_mask=conv_mask))
            feat_in = lcfg['filters']

        # self.featurizer = FeatureFactory.from_config(cfg['input'])
        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def forward(self, audio_signal, length):
        s_input, length = self.encoder(([audio_signal], length))
        return s_input[-1], length


class JasperDecoderForCTC(TrainableNM):
    """
    Jasper Decoder creates the final layer in Jasper that maps from the outputs
    of Jasper Encoder to the vocabulary of interest.

    Args:
        feat_in (int): Number of channels being input to this module
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "encoder_output": NeuralType(
                {0: AxisType(BatchTag),
                 1: AxisType(EncodedRepresentationTag),
                 2: AxisType(ProcessedTimeTag)})}
        output_ports = {
            "output": NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })}
        return input_ports, output_ports

    def __init__(
            self, *,
            feat_in,
            num_classes,
            init_mode="xavier_uniform",
            **kwargs
    ):
        TrainableNM.__init__(self, **kwargs)

        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes,
                      kernel_size=1, bias=True),
            nn.LogSoftmax(dim=1))
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def forward(self, encoder_output):
        return self.decoder_layers(encoder_output).transpose(1, 2)
