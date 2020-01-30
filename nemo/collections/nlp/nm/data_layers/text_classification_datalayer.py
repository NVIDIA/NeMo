from nemo.collections.nlp.data import BertSentenceClassificationDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag


class BertSentenceClassificationDataLayer(TextDataLayer):
    """
    Creates the data layer to use for the task of sentence classification
    with pretrained model.

    All the data processing is done BertSentenceClassificationDataset.

    Args:
        dataset (BertSentenceClassificationDataset):
                the dataset that needs to be converted to DataLayerNM
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels:
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        input_file,
        tokenizer,
        max_seq_length,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        dataset_type=BertSentenceClassificationDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'input_file': input_file,
            'tokenizer': tokenizer,
            'max_seq_length': max_seq_length,
            'num_samples': num_samples,
            'shuffle': shuffle,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)