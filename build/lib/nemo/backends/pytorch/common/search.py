__all__ = ['GreedySearch', 'BeamSearch']

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

INF = float('inf')
BIG_NUM = 1e4


# TODO: Validate, compare to `BeamSearch`
class GreedySearch(NonTrainableNM):
    """Greedy translation search.

    For encoder-decoder based models.

    Args:
        decoder (:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>`):
            Neural module with `.forward_step(...)` function.
        pad_id (int): Label position of padding symbol
        bos_id (int): Label position of start of string symbol
        eos_id (int): Label position of end of string symbol
        max_len (int): Maximum length of sample when doing inference
        batch_size (int): When there is no encoder outputs passed to forward,
            batch size will be used for determine number of samples to infer.
            Defaults to None.

    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # 'encoder_outputs': NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),}, optional=True,
            # )
            "encoder_outputs": NeuralType(('B', 'T', 'D'), ChannelType(), optional=True)
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        """
        return {
            # 'predictions': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # 'attention_weights': NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(TimeTag),}),
            "predictions": NeuralType(('B', 'T'), ChannelType()),
            "attention_weights": NeuralType(('B', 'T', 'T'), ChannelType()),
        }

    def __init__(self, decoder, pad_id, bos_id, eos_id, max_len, batch_size=None):
        super().__init__()

        self.decoder = decoder
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.batch_size = batch_size

    @torch.no_grad()
    def forward(self, encoder_output):
        batch_size = encoder_output.size(0)
        predictions = torch.empty(batch_size, 1, dtype=torch.long, device=self._device).fill_(self.bos_id)
        pad_profile = torch.zeros_like(predictions)

        last_hidden = None
        for i in range(self.max_len):
            log_prob, last_hidden = self.decoder.forward_step(predictions[:, -1:], last_hidden, encoder_output)
            next_pred = torch.argmax(log_prob.squueze(1), dim=-1, keepdim=True)
            # noinspection PyTypeChecker
            next_pred = self.pad_id * pad_profile + next_pred * (1 - pad_profile)
            predictions = torch.cat((predictions, next_pred), dim=-1)
            pad_profile = torch.max(pad_profile, (next_pred == self.eos_id).long())

            if pad_profile.sum() == batch_size:
                break

        return predictions


class BeamSearch(GreedySearch):
    """Beam translation search.

    For encoder-decoder based models.

    Args:
        decoder (:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>`):
            Neural module with `.forward_step(...)` function.
        pad_id (int): Label position of padding symbol
        bos_id (int): Label position of start of string symbol
        eos_id (int): Label position of end of string symbol
        max_len (int): Maximum length of sample when doing inference
        batch_size (int): When there is no encoder outputs passed to forward,
            batch size will be used for determine number of samples to infer.
            Defaults to None.
        beam_size (int): Number of beams (e.g. candidates) to generate and keep
            while doing inference.
            Defaults to 8.

    """

    def __init__(self, decoder, pad_id, bos_id, eos_id, max_len, batch_size=None, beam_size=8):
        super().__init__(decoder, pad_id, bos_id, eos_id, max_len, batch_size)

        self.beam_size = beam_size

    def forward(self, encoder_outputs=None):
        k = self.beam_size
        fdtype = self.decoder.embedding.weight.dtype
        if self.batch_size is None:
            encoder_outputs = encoder_outputs.to(fdtype)
            bs = encoder_outputs.size(0)
            # [BK]TC
            # encoder_output = encoder_output.repeat_interleave(k, 0)
            encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, k, 1, 1)
            encoder_outputs = encoder_outputs.view(-1, *encoder_outputs.shape[2:])
        else:
            bs = self.batch_size

        predictions = torch.empty(bs * k, 1, dtype=torch.long, device=self._device).fill_(self.bos_id)  # [BK]1
        scores = torch.zeros_like(predictions, dtype=fdtype)  # [BK]1
        pad_profile = torch.zeros_like(predictions)  # [BK]1
        if encoder_outputs is not None:
            t = encoder_outputs.shape[1]
            # [BK]1T
            attention_weights = torch.empty(bs * k, 1, t, dtype=fdtype, device=self._device).fill_(1.0 / t)
        else:
            attention_weights = None

        last_hidden = None
        for i in range(self.max_len):
            (log_probs, last_hidden, attention_weights_i,) = self.decoder.forward_step(
                predictions[:, -1:], encoder_outputs, last_hidden
            )  # [BK]1C, h[BK]C, [BK]1T

            log_probs = log_probs.squeeze(1)  # [BK]C

            # We need to strike out other (k - 1) options because they equal
            # to the first one. And also, take care of NaN's and INF's.
            log_probs.clamp_(-BIG_NUM, BIG_NUM)
            if i == 0:
                log_probs.view(bs, k, -1)[:, 1:, :] = -INF
            log_probs[torch.isnan(log_probs)] = -INF

            scores_i, predicted_i = log_probs.topk(k)  # [BK]K

            # noinspection PyTypeChecker
            mask = pad_profile.squeeze().byte()
            scores_i[mask, 0] = 0.0
            scores = scores + scores_i
            scores[mask, 1:] = -INF
            scores, indices_i = torch.topk(scores.view(-1, k ** 2), k)  # BK, BK
            scores = scores.view(-1, 1)  # [BK]1

            pad_mask = pad_profile.repeat(1, k)  # [BK]K
            # noinspection PyTypeChecker
            # [BK]K
            predicted_i = pad_mask * self.pad_id + (1 - pad_mask) * predicted_i
            predictions = predictions.unsqueeze(1).repeat(1, k, 1)  # [BK]KL
            # [BK]K[L+1]
            predictions = torch.cat((predictions, predicted_i.unsqueeze(2)), dim=-1)
            predictions = (
                predictions.view(bs, k ** 2, -1)
                .gather(1, indices_i.unsqueeze(2).repeat(1, 1, predictions.size(-1)),)
                .view(-1, predictions.size(-1))
            )  # [BK][L+1]

            new_tensors = []
            for t in last_hidden:
                new_tensors.append(self.choose(t, indices_i, 1))
            last_hidden = tuple(new_tensors)

            if attention_weights_i is not None:
                attention_weights = torch.cat((attention_weights, attention_weights_i), dim=1)
                attention_weights = self.choose(attention_weights, indices_i, 0)

            pad_profile = ((predictions[:, -1:] == self.eos_id) | (predictions[:, -1:] == self.pad_id)).long()  # [BK]1

            if pad_profile.sum() == bs * k:
                break

        best_i = torch.argmax(scores.view(bs, k), dim=-1, keepdim=True)  # B1
        predictions = (
            predictions.view(bs, k, -1).gather(1, best_i.repeat(1, predictions.size(1)).unsqueeze(1)).squeeze(1)
        )  # BT
        attention_weights = attention_weights[:, 1:, :]  # -eos
        shape_suf = attention_weights.shape[1:]
        attention_weights = attention_weights.view(bs, k, *shape_suf)
        attention_weights = attention_weights.gather(
            1, best_i.unsqueeze(1).repeat(1, *shape_suf).unsqueeze(1)
        ).squeeze(1)

        return predictions, attention_weights

    @staticmethod
    def choose(tensor, indices, dim):
        """(*[BK]*, BK, int) -> *[BK]*"""

        bs, k = indices.shape
        indices = torch.div(indices, k)
        shift = torch.arange(bs, device=indices.device)[:, None] * k
        indices = (indices + shift).view(-1)
        return tensor.index_select(dim, indices)
