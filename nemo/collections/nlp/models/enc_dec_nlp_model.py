from nemo.collections.nlp.models.nlp_model import NLPModel

__all__ = ['NLPModel']


class EncDecNLPModel(NLPModel):
    """Base class for encoder-decoder NLP models.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "decoder_input_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "loss": NeuralType((), LossType()),
            "decoder_hidden_states": NeuralType(("B", "T", "D"), ChannelType(), optional=True),
            "encoder_hidden_states": NeuralType(("B", "T", "D"), ChannelType(), optional=True),
        }
