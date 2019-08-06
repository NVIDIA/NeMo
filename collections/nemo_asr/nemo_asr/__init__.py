# Copyright (c) 2019 NVIDIA Corporation
from nemo.core import Backend

from .data_layer import AudioToTextDataLayer, AudioPreprocessing, \
    SpectrogramAugmentation, MultiplyBatch
from .greedy_ctc_decoder import GreedyCTCDecoder
from .beam_search_decoder import BeamSearchDecoderWithLM
from .jasper import JasperEncoder, JasperDecoderForCTC
from .las.misc import JasperRNNConnector
from .losses import CTCLossNM

name = "nemo_asr"
backend = Backend.PyTorch
