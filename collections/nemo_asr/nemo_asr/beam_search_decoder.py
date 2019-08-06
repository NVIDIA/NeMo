# Copyright (c) 2019 NVIDIA Corporation
# Requires Baidu's CTC decoders from
# https://github.com/PaddlePaddle/DeepSpeech/decoders/swig

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core import DeviceType
from nemo.core.neural_types import *


class BeamSearchDecoderWithLM(NonTrainableNM):
    """Neural Module that does CTC beam search with a n-gram language model.

    It takes a batch of log_probabilities. Note the bigger the batch, the
    better as proccessing is parallelized. Outputs a list of size batch_size.
    Each element in the list is a list of size beam_search, and each element
    in that list is a tuple of (final_log_prob, hyp_string).

    Args:
        vocab (list): List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        beam_width (int): Size of beams to keep and expand upon. Larger beams
            result in more accurate but slower predictions
        alpha (float): The amount of importance to place on the n-gram language
            model. Larger alpha means more importance on the LM and less
            importance on the acoustic model (Jasper).
        beta (float): A penalty term given to longer word sequences. Larger
            beta will result in shorter sequences.
        lm_path (str): Path to n-gram language model
        num_cpus (int): Number of cpus to use
    """

    @staticmethod
    def create_ports():
        input_ports = {
            "log_probs": NeuralType({0: AxisType(BatchTag),
                                     1: AxisType(TimeTag),
                                     2: AxisType(ChannelTag)}),
            "log_probs_length": NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "predictions": NeuralType(None)
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            vocab,
            beam_width,
            alpha,
            beta,
            lm_path,
            num_cpus,
            **kwargs):

        try:
            from ctc_decoders import Scorer
            from ctc_decoders import ctc_beam_search_decoder_batch
        except ModuleNotFoundError:
            raise ModuleNotFoundError("BeamSearchDecoderWithLM requires the "
                                      "installation of ctc_decoders "
                                      "from nemo/scripts/install_decoders.py")

        self.scorer = Scorer(
            alpha,
            beta,
            model_path=lm_path,
            vocabulary=vocab
        )
        self.beam_search_func = ctc_beam_search_decoder_batch

        super().__init__(
            # Override default placement from neural factory
            placement=DeviceType.CPU,
            **kwargs)

        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus

    def forward(self, log_probs, log_probs_length):
        probs = torch.exp(log_probs)
        probs_list = []
        for i, prob in enumerate(probs):
            probs_list.append(prob[:log_probs_length[i], :])
        res = self.beam_search_func(
            probs_list,
            self.vocab,
            beam_size=self.beam_width,
            num_processes=self.num_cpus,
            ext_scoring_func=self.scorer,
        )
        return [res]
