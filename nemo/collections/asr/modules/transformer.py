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


from nemo.collections.common.parts import NEG_INF, mask_padded_tokens

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

        input_ids = torch.transpose(input_ids, 1, 2)  # B D T -> B T D

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

        self._output_layer = torch.nn.Linear(hidden_size, vocab_size)

    @typecheck()
    def forward(self, input_ids, encoder_embeddings, encoder_mask):
        """
        input_ids:  "input_ids": NeuralType(('B', 'T'), ChannelType()),
        encoder_embeddings:  "encoder_embeddings": NeuralType(('B', 'T', 'D'), ChannelType(), optional=True),
        encoder_mask:  "encoder_mask": NeuralType(('B', 'T'), MaskType(), optional=True),

        """ 
        # prepare pad and mask for output seq of labels
        ys_in_pad, ys_out_pad = add_sos_eos(input_ids) # (B, L_dec+1)

        ys_mask = target_mask(ys_in_pad) #(B, L_dec+1)

        decoder_embeddings = self._embedding(input_ids=ys_in_pad)
        
        decoder_hidden_states = self._decoder(
            decoder_states=decoder_embeddings,  # output of the embedding layer (B x L_dec x H)
            decoder_mask=ys_mask,  # decoder inputs mask (B x L_dec)
            encoder_states=encoder_embeddings,  # output of the encoder (B x L_enc x H)
            encoder_mask=encoder_mask, # encoder inputs mask (B x L_enc)
        )

        logits = self._output_layer(decoder_hidden_states)
        return logits, ys_out_pad



    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
    ):
        """
        One step of autoregressive output generation.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding

        input_ids: B D T
        decoder_mask:  decoder inputs mask (B x L_dec)
        encoder_embeddings: (B x L_enc x H)
        """

        # decoder_hidden_states = self.embedding.forward(decoder_input_ids, start_pos=pos)
        decoder_hidden_states = self._embedding(input_ids=decoder_input_ids)
        # self.pad = 4
        # decoder_input_mask = mask_padded_tokens(decoder_input_ids, self.pad).float()\
       
        decoder_input_mask = target_mask(decoder_input_ids) #(B, L_dec+1)

        if encoder_hidden_states is not None:
            decoder_mems_list = self._decoder.forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=True,
            )
        else:
            decoder_mems_list = self._decoder.forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=True
            )

        decoder_hidden_states = decoder_mems_list[-1][:, -1:]
        # decoder_hidden_states = decoder_mems_list[-1]
        logits = self._output_layer(decoder_hidden_states)
        log_probs = torch.nn.functional.log_softmax(logits[:, -1:], dim=2)
        return log_probs, decoder_mems_list

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
