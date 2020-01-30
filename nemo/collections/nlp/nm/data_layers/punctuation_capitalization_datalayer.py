from nemo.collections.nlp.data import BertPunctuationCapitalizationDataset, BertTokenClassificationInferDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import NeuralType, AxisType, BatchTag, TimeTag


class BertPunctuationCapitalizationDataLayer(TextDataLayer):
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

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        punct_labels:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        capit_labels:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "punct_labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "capit_labels": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        text_file,
        label_file,
        tokenizer,
        max_seq_length,
        pad_label='O',
        punct_label_ids=None,
        capit_label_ids=None,
        num_samples=-1,
        shuffle=False,
        batch_size=64,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        dataset_type=BertPunctuationCapitalizationDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {
            'text_file': text_file,
            'label_file': label_file,
            'max_seq_length': max_seq_length,
            'tokenizer': tokenizer,
            'num_samples': num_samples,
            'shuffle': shuffle,
            'pad_label': pad_label,
            'punct_label_ids': punct_label_ids,
            'capit_label_ids': capit_label_ids,
            'ignore_extra_tokens': ignore_extra_tokens,
            'ignore_start_end': ignore_start_end,
            'use_cache': use_cache,
        }
        super().__init__(dataset_type, dataset_params, **kwargs)


class BertPunctuationCapitalizationInferDataLayer(TextDataLayer):
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

        loss_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        subtokens_mask:
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "loss_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "subtokens_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
        }

    def __init__(
        self,
        queries,
        tokenizer,
        max_seq_length,
        batch_size=1,
        dataset_type=BertTokenClassificationInferDataset,
        **kwargs
    ):
        kwargs['batch_size'] = batch_size
        dataset_params = {'queries': queries, 'tokenizer': tokenizer, 'max_seq_length': max_seq_length}
        super().__init__(dataset_type, dataset_params, **kwargs)