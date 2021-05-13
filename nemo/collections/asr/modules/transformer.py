from collections import OrderedDict
from typing import Any, Dict, Optional

from nemo.collections.asr.parts.transformer_utils import *
from nemo.collections.nlp.modules.common.decoder_module import DecoderModule
from nemo.collections.nlp.modules.common.encoder_module import EncoderModule
from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.modules.common.transformer.transformer_modules import (
    TransformerASREmbedding,
    TransformerEmbedding,
)
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, MaskType, NeuralType, SpectrogramType


class TransformerEncoderNM(EncoderModule, Exportable):
    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "input_ids": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return OrderedDict(
            {
                "last_hidden_states": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation()),
                "length": NeuralType((tuple('B')), LengthsType()),
                "encoder_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            }
        )

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        mask_future: bool = False,
        pre_ln: bool = False,
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size

        self._embedding = TransformerASREmbedding(idim=32, input_layer='linear')

        self._encoder = TransformerEncoder(
            hidden_size=self._hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            mask_future=mask_future,
            pre_ln=pre_ln,
        )

    @typecheck()
    def forward(self, input_ids, length=None):

        """
        input_ids :B  D  T/L_enc
        encoder_hidden_states: B T/L_enc H
        
        """

        if length is None:
            length = torch.tensor(input_ids.size(-1)).repeat(input_ids.size(0)).to(input_ids)

        input_ids = torch.transpose(input_ids, 1, 2)  # B T D

        bs, xmax, idim = input_ids.size()
        # Create src mask
        src_mask = make_pad_mask(length, max_time=xmax, device=input_ids.device)

        # form embedding
        embeddings = self._embedding(input_ids=input_ids)
        # encode with TransformerEncoder
        encoder_hidden_states = self._encoder(encoder_states=embeddings, encoder_mask=src_mask)

        return encoder_hidden_states, length, src_mask  # return src_mask for transformer decoder

    @property
    def hidden_size(self):
        return self._hidden_size

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(2, 16), device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        return tuple([input_ids, encoder_mask])


class TransformerDecoderNM(DecoderModule, Exportable):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        inner_size: int,
        num_attention_heads: int,
        max_sequence_length: int = 512,
        num_token_types: int = 2,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
        ffn_dropout: float = 0.0,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        hidden_act: str = 'relu',
        pre_ln: bool = False,
        use_output_layer=False,
        restricted: int = -1
    ):
        super().__init__()

        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._max_sequence_length = max_sequence_length

        self._embedding = TransformerASREmbedding(idim=vocab_size, input_layer='embed')

        self._decoder = TransformerDecoder(
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            ffn_dropout=ffn_dropout,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            restricted=restricted
        )

        # self._after_norm = torch.nn.LayerNorm(hidden_size, eps=1e-5)

        self._output_layer = torch.nn.Linear(hidden_size, vocab_size)

    @typecheck()
    def forward(self, input_ids, encoder_embeddings, encoder_mask):
        # prepare pad and mask for output seq of labels
        ys_in_pad, ys_out_pad = add_sos_eos(input_ids)
        ys_mask = target_mask(ys_in_pad)

        decoder_embeddings = self._embedding(input_ids=ys_in_pad)

        decoder_hidden_states = self._decoder(
            decoder_states=decoder_embeddings,  # output of the embedding layer (B x L_dec x H)
            decoder_mask=ys_mask,  # decoder inputs mask (B x L_dec)
            encoder_states=encoder_embeddings,  # output of the encoder (B x L_enc x H)
            encoder_mask=encoder_mask, # encoder inputs mask (B x L_enc)
        )
        # decoder_hidden_states = self._after_norm(decoder_hidden_states)

        logits = self._output_layer(decoder_hidden_states)
        return logits, ys_out_pad

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def embedding(self):
        return self._embedding

    @property
    def decoder(self):
        return self._decoder

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(2, 16), device=sample.device)
        encoder_mask = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        return tuple([input_ids, encoder_mask, self._embedding(input_ids), encoder_mask])
