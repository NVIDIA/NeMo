import torch

from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types import LengthsType, LogitsType, LogprobsType, NeuralType, PredictionsType
from nemo.utils import logging


class BeamSearchDecoderWithLM(NeuralModule):
    """Neural Module that does CTC beam search with a n-gram language model.
    It takes a batch of log_probabilities. Note the bigger the batch, the
    better as processing is parallelized. Outputs a list of size batch_size.
    Each element in the list is a list of size beam_search, and each element
    in that list is a tuple of (final_log_prob, hyp_string).
    Args:
        vocab (list): List of characters that can be output by the ASR model. For Jasper, this is the 28 character set
            {a-z '}. The CTC blank symbol is automatically added later for models using ctc.
        beam_width (int): Size of beams to keep and expand upon. Larger beams result in more accurate but slower
            predictions
        alpha (float): The amount of importance to place on the n-gram language model. Larger alpha means more
            importance on the LM and less importance on the acoustic model (Jasper).
        beta (float): A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
        lm_path (str): Path to n-gram language model
        num_cpus (int): Number of cpus to use
        cutoff_prob (float): Cutoff probability in vocabulary pruning, default 1.0, no pruning
        cutoff_top_n (int): Cutoff number in pruning, only top cutoff_top_n characters with highest probs in
            vocabulary will be used in beam search, default 40.
        input_tensor (bool): Set to True if you intend to pass PyTorch Tensors, set to False if you intend to pass
            numpy arrays.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "log_probs_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(
        self, vocab, beam_width, alpha, beta, lm_path, num_cpus, cutoff_prob=1.0, cutoff_top_n=40, input_tensor=False
    ):

        try:
            from ctc_decoders import Scorer
            from ctc_decoders import ctc_beam_search_decoder_batch
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "BeamSearchDecoderWithLM requires the "
                "installation of ctc_decoders "
                "from scripts/install_ctc_decoders.sh"
            )

        super().__init__()

        if lm_path is not None:
            self.scorer = Scorer(alpha, beta, model_path=lm_path, vocabulary=vocab)
        else:
            self.scorer = None
        self.beam_search_func = ctc_beam_search_decoder_batch
        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.input_tensor = input_tensor

    @typecheck()
    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        probs_list = log_probs
        if self.input_tensor:
            probs = torch.exp(log_probs)
            probs_list = []
            for i, prob in enumerate(probs):
                probs_list.append(prob[: log_probs_length[i], :])
        res = self.beam_search_func(
            probs_list,
            self.vocab,
            beam_size=self.beam_width,
            num_processes=self.num_cpus,
            ext_scoring_func=self.scorer,
            cutoff_prob=self.cutoff_prob,
            cutoff_top_n=self.cutoff_top_n,
        )
        return res
