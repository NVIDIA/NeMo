# Copyright (c) 2019 NVIDIA Corporation
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import nemo
from .parts.jasper import JasperBlock, init_weights, jasper_activations
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

logging = nemo.logging


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
                    'se' (bool)  # Whether to add Squeeze and Excitation
                        # sub-blocks.
                        # Defaults to False
                    'se_reduction_ratio' (int)  # The reduction ratio of the Squeeze
                        # sub-module.
                        # Must be an integer > 1.
                        # Defaults to 16
                    'kernel_size_factor' (float)  # Conv kernel size multiplier
                        # Can be either an int or float
                        # Kernel size is recomputed as below:
                        # new_kernel_size = int(max(1, (kernel_size * kernel_width)))
                        # to prevent kernel sizes than 1.
                        # Note: If rescaled kernel size is an even integer,
                        # adds 1 to the rescaled kernel size to allow "same"
                        # padding.
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

    length: Optional[torch.Tensor]

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "audio_signal": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "outputs": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(EncodedRepresentationTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "encoded_lengths": NeuralType({0: AxisType(BatchTag)}),
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def _disabled_deployment_input_ports(self):
        return set(["length"])

    @property
    def _disabled_deployment_output_ports(self):
        return set(["encoded_lengths"])

    def _prepare_for_deployment(self):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        logging.warning(f"Turned off {m_count} masked convolutions")

    def __init__(
        self,
        jasper,
        activation,
        feat_in,
        normalization_mode="batch",
        residual_mode="add",
        norm_groups=-1,
        conv_mask=True,
        frame_splicing=1,
        init_mode='xavier_uniform',
    ):
        super().__init__()

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
            heads = lcfg.get('heads', -1)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 16)
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
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
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    kernel_size_factor=kernel_size_factor,
                )
            )
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def forward(self, audio_signal, length=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor, Optional[Tensor]

        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]
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

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "encoder_output": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(EncodedRepresentationTag), 2: AxisType(ProcessedTimeTag),}
            # )
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"output": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),})}
        return {"output": NeuralType(('B', 'T', 'D'), LogprobsType())}

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform"):
        super().__init__()

        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = nn.Sequential(nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def forward(self, encoder_output):
        return F.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)


class JasperDecoderForClassification(TrainableNM):
    """
        Jasper Decoder creates the final layer in Jasper that maps from the outputs
        of Jasper Encoder to one class label.

        Args:
            feat_in (int): Number of channels being input to this module
            num_classes (int): Number of characters in ASR model's vocab/labels.
                This count should not include the CTC blank symbol.
            init_mode (str): Describes how neural network parameters are
                initialized. Options are ['xavier_uniform', 'xavier_normal',
                'kaiming_uniform','kaiming_normal'].
                Defaults to "xavier_uniform".
        """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "encoder_output": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(EncodedRepresentationTag), 2: AxisType(ProcessedTimeTag)}
            # )
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"logits": NeuralType({0: AxisType(BatchTag), 1: AxisType(ChannelTag)})}
        return {"logits": NeuralType(('B', 'D'), LogitsType())}

    def __init__(
        self, *, feat_in, num_classes, init_mode="xavier_uniform", return_logits=True, pooling_type='avg', **kwargs
    ):
        TrainableNM.__init__(self, **kwargs)

        self._feat_in = feat_in
        self._return_logits = return_logits
        self._num_classes = num_classes

        if pooling_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError('Pooling type chosen is not valid. Must be either `avg` or `max`')

        self.decoder_layers = nn.Sequential(nn.Linear(self._feat_in, self._num_classes, bias=True))
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.to(self._device)

    def forward(self, encoder_output):
        batch, in_channels, timesteps = encoder_output.size()

        encoder_output = self.pooling(encoder_output).view(batch, in_channels)  # [B, C]
        logits = self.decoder_layers(encoder_output)  # [B, num_classes]

        if self._return_logits:
            return logits

        return F.softmax(logits, dim=-1)
