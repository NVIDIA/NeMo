from nemo.collections.nlp.data import LanguageModelingDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag


class LanguageModelingDataLayer(TextDataLayer):
    """
    Data layer for standard language modeling task.

    Args:
        dataset (str): path to text document with data
        tokenizer (TokenizerSpec): tokenizer
        max_seq_length (int): maximum allowed length of the text segments
        batch_step (int): how many tokens to skip between two successive
            segments of text when constructing batches
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of tokens which should be predicted from each of the
            corresponding tokens in input_ids; for left-to-right language
            modeling equals to input_ids shifted by 1 to the right
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)
        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self, dataset, tokenizer, max_seq_length, batch_step=128, dataset_type=LanguageModelingDataset, **kwargs
    ):
        dataset_params = {
            'dataset': dataset,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'batch_step': batch_step,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)
