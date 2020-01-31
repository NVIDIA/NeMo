__all__ = ['BertQuestionAnsweringDataLayer']
from nemo.collections.nlp.data import SquadDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag


class BertQuestionAnsweringDataLayer(TextDataLayer):
    """
    Creates the data layer to use for Question Answering classification task.

    Args:
        data_dir (str): Directory that contains train.*.json and dev.*.json.
        tokenizer (obj): Tokenizer object, e.g. NemoBertTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (iny): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        mode (str): Use "train" or "dev" to define between
            training and evaluation.
        batch_size (int): Batch size. Defaults to 64.
        dataset_type (class): Question Answering class.
            Defaults to SquadDataset.
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

            start_positions:
                0: AxisType(BatchTag)

            end_positions:
                0: AxisType(BatchTag)

            unique_ids:
                0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "start_positions": NeuralType({0: AxisType(BatchTag)}),
            "end_positions": NeuralType({0: AxisType(BatchTag)}),
            "unique_ids": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(
        self,
        data_dir,
        tokenizer,
        version_2_with_negative,
        doc_stride,
        max_query_length,
        max_seq_length,
        mode="train",
        batch_size=64,
        dataset_type=SquadDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'data_dir': data_dir,
            'mode': mode,
            'tokenizer': tokenizer,
            'version_2_with_negative': version_2_with_negative,
            'max_query_length': max_query_length,
            'max_seq_length': max_seq_length,
            'doc_stride': doc_stride,
        }

        super().__init__(dataset_type, dataset_params, **kwargs)
